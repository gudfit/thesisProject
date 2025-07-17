# src/core/data_handler.py
from datasets import load_dataset
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    @staticmethod
    def load_sentences(dataset_name: str, subset: Optional[str], split: str, max_samples: Optional[int] = None) -> List[str]:
        if subset is None or subset == "" or subset == "default":
            ds = load_dataset(dataset_name, split=split)
        else:
            ds = load_dataset(dataset_name, subset, split=split)
        cols = ds.column_names
        if "text" in cols:
            texts = ds["text"]
        elif "sentence" in cols:
            texts = ds["sentence"]
        elif "content" in cols:
            texts = ds["content"]
        elif "description" in cols:
            texts = ds["description"]
        else:
            col0 = cols[0]
            texts = ds[col0]
        sents = [str(t).strip() for t in texts if t and str(t).strip()]
        if max_samples:
            sents = sents[:max_samples]
        return sents

    @staticmethod
    def prepare_tokenizer(tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

