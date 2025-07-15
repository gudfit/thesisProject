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
        self.model = None
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self):
        if self.model is None:
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)

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

    @abstractmethod
    def fine_tune(
        self,
        train_texts: List[str],
        eval_texts: List[str],
        **kwargs,
    ):
        pass

    def save_model(self, path: str):
        if self.model:
            self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path: str):
        self.model = AutoModelForMaskedLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
