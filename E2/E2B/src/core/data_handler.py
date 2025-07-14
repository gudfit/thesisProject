from datasets import load_dataset
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataHandler:
    @staticmethod
    def load_sentences(
        dataset_name: str, subset: str, split: str, max_samples: int = None
    ) -> List[str]:
        logger.info(f"Loading dataset: {dataset_name}/{subset} split: {split}")
        dataset = load_dataset(dataset_name, subset, split=split)
        if "text" in dataset.column_names:
            texts = dataset["text"]
        elif "sentence" in dataset.column_names:
            texts = dataset["sentence"]
        else:
            raise ValueError(f"Cannot find text column in dataset {dataset_name}")
        sentences = [text.strip() for text in texts if text and text.strip()]
        if max_samples:
            sentences = sentences[:max_samples]

        logger.info(f"Loaded {len(sentences)} sentences")
        return sentences

    @staticmethod
    def prepare_tokenizer(tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
