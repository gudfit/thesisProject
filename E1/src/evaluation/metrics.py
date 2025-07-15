# E1/src/evaluation/metrics.py
"""Evaluation metrics for compression quality."""

import torch
import numpy as np
from typing import Dict
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein
import collections

for res in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{res}")
    except LookupError:
        nltk.download(res, quiet=True)


class CompressionMetrics:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        self.semantic_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.semantic_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
        self.bert_scorer = BERTScorer(lang="en", model_type='roberta-large', device=device)

    def calculate_all_metrics(
        self,
        original_text: str,
        reconstructed_text: str,
        compressed_data: Dict[str, any],
        compressor,
    ) -> Dict[str, float]:
        m: Dict[str, float] = {}
        m["exact_match"] = float(original_text.strip() == reconstructed_text.strip())
        m["character_accuracy"] = self.character_accuracy(
            original_text, reconstructed_text
        )
        m["word_accuracy"] = self.word_accuracy(original_text, reconstructed_text)
        m["levenshtein_distance"] = self.levenshtein_distance(
            original_text, reconstructed_text
        )
        m.update(self.rouge_scores(original_text, reconstructed_text))
        m["bleu_score"] = self.bleu_score(original_text, reconstructed_text)
        m.update(self.bert_score(original_text, reconstructed_text))
        m["semantic_similarity"] = self.semantic_similarity(
            original_text, reconstructed_text
        )
        m["compression_ratio"] = self.compression_ratio(
            original_text, compressed_data, compressor
        )
        m["bits_per_character"] = self.bits_per_character(
            original_text, compressed_data, compressor
        )
        return m

    @staticmethod
    def character_accuracy(original: str, reconstructed: str) -> float:
        if not original:
            return 1.0 if not reconstructed else 0.0
        matches = sum(1 for o, r in zip(original, reconstructed) if o == r)
        return matches / max(len(original), len(reconstructed))

    @staticmethod
    def word_accuracy(original: str, reconstructed: str) -> float:
        original_words = original.lower().split()
        reconstructed_words = reconstructed.lower().split()
        if not original_words:
            return 1.0 if not reconstructed_words else 0.0

        original_counts = collections.Counter(original_words)
        reconstructed_counts = collections.Counter(reconstructed_words)

        common_words_counts = original_counts & reconstructed_counts
        correct_words = sum(common_words_counts.values())

        return correct_words / len(original_words)

    @staticmethod
    def levenshtein_distance(original: str, reconstructed: str) -> float:
        if not original and not reconstructed:
            return 0.0
        return Levenshtein.distance(original, reconstructed) / max(
            len(original), len(reconstructed), 1
        )

    def rouge_scores(self, original: str, reconstructed: str) -> Dict[str, float]:
        s = self.rouge_scorer.score(original, reconstructed)
        return {
            "rouge1_precision": s["rouge1"].precision,
            "rouge1_recall": s["rouge1"].recall,
            "rouge1_fmeasure": s["rouge1"].fmeasure,
            "rouge2_precision": s["rouge2"].precision,
            "rouge2_recall": s["rouge2"].recall,
            "rouge2_fmeasure": s["rouge2"].fmeasure,
            "rougeL_precision": s["rougeL"].precision,
            "rougeL_recall": s["rougeL"].recall,
            "rougeL_fmeasure": s["rougeL"].fmeasure,
        }

    @staticmethod
    def bleu_score(original: str, reconstructed: str) -> float:
        o_tokens = nltk.word_tokenize(original.lower())
        r_tokens = nltk.word_tokenize(reconstructed.lower())
        if not o_tokens:
            return 1.0 if not r_tokens else 0.0
        return sentence_bleu(
            [o_tokens], r_tokens, smoothing_function=SmoothingFunction().method4
        )

    def bert_score(self, original: str, reconstructed: str) -> Dict[str, float]:
        if not original.strip() or not reconstructed.strip():
            return {
                "bert_score_precision": 0.0,
                "bert_score_recall": 0.0,
                "bert_score_f1": 0.0,
            }
        P, R, F1 = self.bert_scorer.score([reconstructed], [original])
        return {
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item(),
            "bert_score_f1": F1.mean().item(),
        }

    def semantic_similarity(self, original: str, reconstructed: str) -> float:
        if not original.strip() or not reconstructed.strip():
            return 0.0
        with torch.no_grad():
            o_enc = self.semantic_tokenizer(
                original,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self.device)
            r_enc = self.semantic_tokenizer(
                reconstructed,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self.device)
            o_emb = self._mean_pool(
                self.semantic_model(**o_enc), o_enc["attention_mask"]
            )
            r_emb = self._mean_pool(
                self.semantic_model(**r_enc), r_enc["attention_mask"]
            )
        return float(cosine_similarity(o_emb.cpu().numpy(), r_emb.cpu().numpy())[0, 0])

    @staticmethod
    def _mean_pool(model_output, attention_mask):
        t = model_output.last_hidden_state
        m = attention_mask.unsqueeze(-1).expand(t.size()).float()
        return torch.sum(t * m, 1) / torch.clamp(m.sum(1), min=1e-9)

    @staticmethod
    def compression_ratio(
        original_text: str, compressed_data: Dict[str, any], compressor
    ) -> float:
        orig_bits = len(original_text.encode("utf-8")) * 8
        comp_bits = compressor._calculate_compressed_size(compressed_data)
        return orig_bits / comp_bits if comp_bits else float("inf")

    @staticmethod
    def bits_per_character(
        original_text: str, compressed_data: Dict[str, any], compressor
    ) -> float:
        if not original_text:
            return 0.0
        comp_bits = compressor._calculate_compressed_size(compressed_data)
        return comp_bits / len(original_text)

    def perplexity(self, model, tokenizer, text: str) -> float:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            loss = model(
                enc["input_ids"].to(self.device),
                labels=enc["input_ids"].to(self.device),
            ).loss
        return torch.exp(loss).item()
