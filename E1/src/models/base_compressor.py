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
        masking_probability: float = 0.15,
    ):
        lr = float(learning_rate)

        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=128):
                self.tokenizer = tokenizer
                self.texts = texts
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                enc = self.tokenizer(
                    self.texts[idx],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                return {k: v.squeeze() for k, v in enc.items()}

        train_dataset = TextDataset(train_texts, self.tokenizer)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        steps = epochs * len(train_loader)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=steps,
        )
        metrics_calc = CompressionMetrics(device=self.device)
        pb = tqdm(range(steps), desc="Fine‑tuning")
        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                inputs = batch["input_ids"].to(self.device)
                attn = batch["attention_mask"].to(self.device)
                labels = inputs.clone()
                rand = torch.rand(inputs.shape, device=self.device)
                mask_arr = (rand < masking_probability) & (
                    inputs != self.tokenizer.pad_token_id
                )
                inputs[mask_arr] = self.tokenizer.mask_token_id
                labels[~mask_arr] = -100
                loss = self.model(
                    input_ids=inputs, attention_mask=attn, labels=labels
                ).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pb.update(1)
            self.model.eval()
            tot_sim, tot_acc, cnt = 0, 0, 0
            for text in eval_texts[:50]:
                try:
                    comp = self.compress(text, masking_probability=masking_probability)
                    recon = self.decompress(comp)
                    sim = metrics_calc.semantic_similarity(text, recon)
                    acc = metrics_calc.word_accuracy(text, recon)
                    tot_sim += sim
                    tot_acc += acc
                    cnt += 1
                except Exception:
                    continue
            avg_sim = tot_sim / cnt if cnt else 0
            avg_acc = tot_acc / cnt if cnt else 0
            print(
                f"Epoch {epoch+1}: mask_prob={masking_probability} | AvgSim={avg_sim:.4f} | AvgWordAcc={avg_acc:.4f}"
            )
            self.model.train()

    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path: str):
        self.model = AutoModelForMaskedLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
