# src/models/model_utils.py
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

class ModelManager:
    @staticmethod
    def load_model_and_tokenizer(model_path: str, device: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model from {model_path} to {device}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        return model, tokenizer

    @staticmethod
    def save_model_and_tokenizer(model, tokenizer, save_path: str):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    @staticmethod
    def get_model_size_on_disk(model_path: str) -> int:
        total_size = 0
        for dirpath, _, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

    @staticmethod
    def count_nonzero_and_total_params(model) -> Dict[str, int]:
        total = 0
        nonzero = 0
        effective_bytes = 0
        for p in model.parameters():
            n = p.numel()
            nz = int(torch.count_nonzero(p).item())
            total += n
            nonzero += nz
            effective_bytes += nz * p.element_size()
        return {
            "total_params": total,
            "nonzero_params": nonzero,
            "effective_param_bytes": effective_bytes,
        }

    @staticmethod
    def count_nonzero_params(model) -> int:
        return sum(torch.sum(p != 0).item() for p in model.parameters())

    @staticmethod
    def cleanup_model(model, tokenizer):
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

