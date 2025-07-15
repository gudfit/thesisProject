# src/models/base_compressor.py
"""Base class for LLM-based compression methods."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import numpy as np
import inspect
from ..evaluation.metrics import CompressionMetrics


class BaseCompressor(ABC):
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @abstractmethod
    def compress(self, text: str, **kwargs) -> Dict[str, any]:
        pass

    @abstractmethod
    def decompress(self, compressed_data: Dict[str, any]) -> str:
        pass

    def calculate_compression_ratio(
        self, original_text: str, compressed_data: Dict[str, any]
    ) -> float:
        original_size = len(original_text.encode("utf-8")) * 8
        compressed_size = self._calculate_compressed_size(compressed_data)
        return original_size / compressed_size if compressed_size > 0 else float("inf")

    @abstractmethod
    def _calculate_compressed_size(self, compressed_data: Dict[str, any]) -> int:
        pass
        
    def fine_tune(
        self,
        train_texts: List[str],
        eval_texts: List[str],
        epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 16,
    ):
        """Custom fine-tuning loop with epoch-wise reconstruction evaluation."""
        lr = float(learning_rate)
        
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=128):
                self.tokenizer = tokenizer
                self.texts = texts
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                return {key: val.squeeze() for key, val in encoding.items()}

        train_dataset = TextDataset(train_texts, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        
        optimizer = AdamW(self.model.parameters(), lr=lr) 
        num_training_steps = epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
        )
        
        metrics_calculator = CompressionMetrics(device=self.device)

        progress_bar = tqdm(range(num_training_steps), desc="Fine-tuning...")

        self.model.train()
        for epoch in range(epochs):
            for batch in train_dataloader:
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = inputs.clone()
                
                prob = 0.15
                rand = torch.rand(inputs.shape, device=self.device)
                mask_arr = (rand < prob) & (inputs != self.tokenizer.pad_token_id) & \
                           (inputs != self.tokenizer.cls_token_id) & (inputs != self.tokenizer.sep_token_id)
                
                # Apply mask
                inputs[mask_arr] = self.tokenizer.mask_token_id
                # Set labels for non-masked tokens to -100 so they are ignored in the loss
                labels[~mask_arr] = -100

                outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # --- Evaluation at the end of each epoch ---
            self.model.eval()
            total_similarity = 0
            total_accuracy = 0
            eval_count = 0
            
            print(f"\n--- Evaluating after Epoch {epoch + 1}/{epochs} ---")
            
            epoch_eval_texts = eval_texts[:50]
            
            with torch.no_grad():
                for text in epoch_eval_texts:
                    try:
                        compressed_data = self.compress(text, masking_probability=0.5)
                        reconstructed_text = self.decompress(compressed_data)
                        
                        similarity = metrics_calculator.semantic_similarity(text, reconstructed_text)
                        accuracy = metrics_calculator.word_accuracy(text, reconstructed_text)
                        
                        total_similarity += similarity
                        total_accuracy += accuracy
                        eval_count += 1
                    except Exception:
                        continue

            avg_similarity = total_similarity / eval_count if eval_count > 0 else 0
            avg_accuracy = total_accuracy / eval_count if eval_count > 0 else 0
            
            print(f"Epoch {epoch + 1} Reconstruction - Avg. Semantic Similarity: {avg_similarity:.4f} | Avg. Word Accuracy: {avg_accuracy:.4f}")
            print("--------------------------------------------------")
            
            self.model.train()


    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path: str):
        self.model = AutoModelForMaskedLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
