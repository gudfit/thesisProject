# src/models/knowledge_compressor.py
"""LLM as Static Knowledge Compressor implementation for Experiment 1B.""" 

import torch
import torch.nn as nn
import numpy as np
import gzip
import os
import json
from typing import Dict, List, Any
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset, Value
from sklearn.metrics import accuracy_score
import time
from tqdm import tqdm
import shutil
import tempfile
import logging

class KnowledgeCompressor:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.logger = logging.getLogger(__name__)

    def fine_tune_on_corpus(self, corpus_texts: List[str], epochs: int = 10, batch_size: int = 8):
        self.logger.info(f"Specializing model by fine-tuning on the target corpus for {epochs} epoch(s)...")
        
        dataset = Dataset.from_dict({"text": corpus_texts})
        tokenized_dataset = dataset.map(
            lambda e: self.tokenizer(e['text'], truncation=True, padding="max_length", max_length=128),
            batched=True, remove_columns=['text']
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        
        training_args = TrainingArguments(
            output_dir="/tmp/corpus_finetune",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_steps=100,
            save_strategy="no",
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        trainer.train()
        self.logger.info("Model specialization complete.")

    def calculate_model_size(self) -> Dict[str, float]:
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        disk_size_bytes = total_params * 4
        return {
            "total_parameters": total_params,
            "disk_size_mb": disk_size_bytes / (1024 * 1024),
        }

    def compress_corpus_gzip(self, texts: List[str]) -> Dict[str, float]:
        full_text = "\n".join(texts).encode("utf-8")
        original_size_mb = len(full_text) / (1024 * 1024)
        compressed_bytes = gzip.compress(full_text, compresslevel=9)
        compressed_size_mb = len(compressed_bytes) / (1024 * 1024)
        return {
            "original_size_mb": original_size_mb,
            "compressed_size_mb": compressed_size_mb,
            "compression_ratio": original_size_mb / compressed_size_mb if compressed_size_mb > 0 else float('inf'),
        }

    def measure_factual_recall(self, factual_probes: List[Dict[str, str]]) -> float:
        correct = 0
        total = len(factual_probes)
        if not total: return 0.0

        for probe in tqdm(factual_probes, desc="Evaluating Factual Recall", leave=False):
            prompt = probe["prompt"]
            expected_answer = probe["answer"].lower().strip()
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
            if mask_token_index.nelement() > 0:
                predicted_id = logits[0, mask_token_index[0]].argmax().item()
                predicted_token = self.tokenizer.decode([predicted_id]).strip().lower()
                if expected_answer in predicted_token or predicted_token in expected_answer:
                    correct += 1
        return correct / total

    def measure_reasoning_capability(self) -> float:
        self.logger.info("Evaluating reasoning by fine-tuning on a subset of MNLI.")
        try:
            dataset = load_dataset("glue", "mnli", split="train[:2000]").train_test_split(test_size=0.5, seed=42)
            train_dataset = dataset['train']
            eval_dataset = dataset['test']

            def tokenize_function(examples):
                return self.tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=128)

            train_dataset = train_dataset.map(tokenize_function, batched=True)
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)

            model_for_nli = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3).to(self.device)

            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                return {"accuracy": accuracy_score(labels, predictions)}

            training_args = TrainingArguments(
                output_dir="/tmp/nli_finetune", num_train_epochs=1, per_device_train_batch_size=16,
                logging_steps=100, save_strategy="no", report_to="none", eval_strategy="epoch"
            )
            trainer = Trainer(
                model=model_for_nli, args=training_args, train_dataset=train_dataset,
                eval_dataset=eval_dataset, compute_metrics=compute_metrics
            )
            
            trainer.train()
            eval_results = trainer.evaluate()
            return eval_results.get("eval_accuracy", 0.0)
        except Exception as e:
            self.logger.error(f"Failed to measure reasoning capability: {e}")
            return 0.0

    def measure_perplexity(self, test_texts: List[str]) -> float:
        self.logger.info("Calculating perplexity using MLM loss on unseen text.")
        
        def to_dataset(texts):
            return Dataset.from_dict({"text": texts})

        test_dataset = to_dataset(test_texts)
        tokenized_dataset = test_dataset.map(lambda e: self.tokenizer(e['text'], truncation=True, padding="max_length", max_length=128), batched=True)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        
        self.model.eval()
        total_loss = 0
        total_batches = 0
        
        data_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=8, collate_fn=data_collator)
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Calculating MLM Loss for Perplexity", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                total_batches += 1
                
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        return np.exp(avg_loss) if avg_loss > 0 else float("inf")

    def measure_inference_cost(self, sample_texts: List[str]) -> Dict[str, float]:
        sample_text = sample_texts[0] if sample_texts else "Sample for benchmark."
        inputs = self.tokenizer(sample_text, return_tensors="pt", max_length=128, truncation=True).to(self.device)
        
        with torch.no_grad():
            for _ in range(10): _ = self.model(**inputs)
        
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = self.model(**inputs)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_inference_time_ms = (end_time - start_time) * 1000 / 50
        return {
            "avg_inference_time_ms": avg_inference_time_ms,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
        }

    def measure_data_efficiency(self, fine_tune_data: List[Dict], target_performance: float) -> Dict[str, Any]:
        self.logger.info(f"Measuring data efficiency to reach target performance of {target_performance:.2f}.")
        data_sizes = [10, 50, 100, 500, 1000]
        results = []
        
        dataset = Dataset.from_list(fine_tune_data).train_test_split(test_size=0.3, seed=42)
        train_ds = dataset['train']
        eval_ds = dataset['test']
        
        num_labels = len(set(train_ds['label']))

        def tokenize(batch):
            return self.tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

        eval_subset = eval_ds.select(range(min(1000, len(eval_ds)))).map(tokenize, batched=True)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {"accuracy": accuracy_score(labels, predictions)}

        for size in data_sizes:
            if size > len(train_ds): break
            self.logger.info(f"Fine-tuning with {size} samples...")
            
            subset_train_ds = train_ds.select(range(size)).map(tokenize, batched=True)
            
            model_for_cls = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels).to(self.device)
            
            training_args = TrainingArguments(
                output_dir="/tmp/kc_data_efficiency", per_device_train_batch_size=8, num_train_epochs=3,
                logging_steps=size, save_strategy="no", report_to="none", eval_strategy="epoch",
                disable_tqdm=True
            )
            trainer = Trainer(
                model=model_for_cls, args=training_args, train_dataset=subset_train_ds, 
                eval_dataset=eval_subset, tokenizer=self.tokenizer, compute_metrics=compute_metrics
            )
            trainer.train()
            
            eval_results = trainer.evaluate()
            accuracy = eval_results.get("eval_accuracy", 0.0)
            results.append({"data_size": size, "performance": accuracy})
            
            if accuracy >= target_performance:
                break
                
        return {"efficiency_curve": results}

    def compare_with_gzip(self, corpus_texts: List[str], test_texts: List[str], factual_probes: List[Dict[str, str]]) -> Dict[str, Any]:
        results = {"llm": {}, "gzip": {}}
        
        self.logger.info("Phase 1: Analyzing sizes and functional utility...")
        results["llm"]["size"] = self.calculate_model_size()
        results["gzip"]["size"] = self.compress_corpus_gzip(corpus_texts)
        results["llm"]["factual_recall"] = self.measure_factual_recall(factual_probes)
        results["llm"]["reasoning_capability"] = self.measure_reasoning_capability()
        results["llm"]["perplexity"] = self.measure_perplexity(test_texts)
        
        self.logger.info("Phase 2: Measuring computational costs...")
        results["llm"]["inference_cost"] = self.measure_inference_cost(test_texts)
        
        self.logger.info("Phase 3: Measuring data efficiency...")
        sst_dataset = load_dataset("glue", "sst2", split="train")
        fine_tune_data = [{"text": ex['sentence'], "label": ex['label']} for ex in sst_dataset]
        results["llm"]["data_efficiency"] = self.measure_data_efficiency(fine_tune_data, target_performance=0.85)
        
        results["comparison"] = {
            "size_ratio": results["llm"]["size"]["disk_size_mb"] / (results["gzip"]["size"]["compressed_size_mb"] + 1e-9)
        }
        return results
