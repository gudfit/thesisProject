# src/models/predictive_masking.py
"""Predictive Masking compression implementation."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_compressor import BaseCompressor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from tqdm.auto import tqdm
import random
import spacy

class PredictiveMaskingCompressor(BaseCompressor):
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        self._load_model()
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.nlp = None

    def _load_spacy(self, model: str = "en_core_web_sm"):
        if self.nlp is None:
            try:
                self.nlp = spacy.load(model)
            except OSError:
                import sys
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", model])
                self.nlp = spacy.load(model)

    @staticmethod
    @torch.no_grad()
    def _token_confidence(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return probs.max(dim=-1).values

    def compress(
        self, text: str, masking_probability: float = 0.5, strategy: str = "intelligent", **kwargs
    ) -> Dict[str, any]:
        self.model.eval()
        should_return_offsets = "intelligent_pos" in strategy
        enc = self.tokenizer(
            text,
            truncation=True,
            return_tensors="pt",
            max_length=512,
            return_offsets_mapping=should_return_offsets,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits

        if strategy == "random":
            mask_positions = self._get_random_mask_positions(enc["input_ids"].squeeze(), masking_probability)
        elif "intelligent" in strategy:
            mask_positions = self._get_intelligent_mask_positions(enc, logits, masking_probability, use_pos=("pos" in strategy))
        else:
            raise ValueError(f"Unknown compression strategy: {strategy}")

        input_ids_squeezed = enc["input_ids"].squeeze()
        if input_ids_squeezed.dim() == 0:
            input_ids_squeezed = input_ids_squeezed.unsqueeze(0)

        unmasked_indices = sorted(list(set(range(len(input_ids_squeezed))) - set(mask_positions)))
        unmasked_token_ids = input_ids_squeezed[unmasked_indices].cpu().numpy()

        return {
            "unmasked_tokens": unmasked_token_ids.astype(np.uint16),
            "unmasked_indices": np.array(unmasked_indices, dtype=np.uint16),
            "original_length": len(input_ids_squeezed),
        }

    def _get_random_mask_positions(self, input_ids: torch.Tensor, masking_probability: float) -> List[int]:
        special_token_ids = self.tokenizer.all_special_ids
        eligible_indices = [
            i for i, token_id in enumerate(input_ids) if token_id not in special_token_ids
        ]
        if not eligible_indices:
            return []
        num_to_mask = int(len(eligible_indices) * masking_probability)
        return sorted(random.sample(eligible_indices, num_to_mask))

    def _get_intelligent_mask_positions(
        self, enc, logits, masking_probability: float, use_pos: bool = False, pos_keep: List[str] = ("NOUN", "VERB", "ADJ")
    ) -> List[int]:
        conf = self._token_confidence(logits)[0]
        input_ids = enc["input_ids"][0]

        if use_pos:
            self._load_spacy()
            doc = self.nlp(self.tokenizer.decode(input_ids, skip_special_tokens=True))
            offsets = enc["offset_mapping"][0].cpu().tolist()
            pos_per_token = self._align_spacy_to_hf(doc, offsets)
            candidate_positions = [i for i, pos in enumerate(pos_per_token) if pos in pos_keep]
            if not candidate_positions:
                 candidate_positions = [i for i, pos in enumerate(pos_per_token) if pos is not None]
        else:
            specials = set(self.tokenizer.all_special_ids)
            candidate_positions = [i for i, tid in enumerate(input_ids.tolist()) if tid not in specials]

        if not candidate_positions:
            return []

        candidate_conf = conf[candidate_positions]
        n_to_mask = max(1, int(masking_probability * len(candidate_positions)))
        topk_indices_in_candidates = torch.topk(candidate_conf, k=min(n_to_mask, len(candidate_conf)), largest=True).indices.tolist()
        positions_to_mask = sorted([candidate_positions[i] for i in topk_indices_in_candidates])

        return positions_to_mask

    def _align_spacy_to_hf(self, doc, offsets: List[Tuple[int, int]]) -> List[Optional[str]]:
        pos_per_token = []
        for (start, end) in offsets:
            if start == end == 0:
                pos_per_token.append(None)
                continue
            midpoint = (start + end) / 2
            found = False
            for spacy_tok in doc:
                if spacy_tok.idx <= midpoint < spacy_tok.idx + len(spacy_tok):
                    pos_per_token.append(spacy_tok.pos_)
                    found = True
                    break
            if not found:
                pos_per_token.append(None)
        return pos_per_token

    def decompress(self, compressed_data: Dict[str, any], num_passes: int = 50, return_ids_for_debug: bool = False):
        self.model.eval()
        original_length = compressed_data["original_length"]
        unmasked_tokens = compressed_data["unmasked_tokens"]
        unmasked_indices = compressed_data["unmasked_indices"]

        decompress_ids = torch.full((1, original_length), self.mask_token_id, dtype=torch.long, device=self.device)
        decompress_ids[0, unmasked_indices] = torch.tensor(unmasked_tokens, dtype=torch.long, device=self.device)

        special_token_ids = self.tokenizer.all_special_ids

        for _ in range(num_passes):
            mask_positions = (decompress_ids == self.mask_token_id).nonzero(as_tuple=False)
            if mask_positions.numel() == 0:
                break

            with torch.no_grad():
                logits = self.model(input_ids=decompress_ids).logits

            mask_confidences = self._token_confidence(logits)[decompress_ids == self.mask_token_id]
            if mask_confidences.numel() == 0:
                break

            best_mask_idx_in_flat_list = mask_confidences.argmax()
            best_position_to_fill = mask_positions[best_mask_idx_in_flat_list]

            token_logits = logits[best_position_to_fill[0], best_position_to_fill[1]]
            token_logits[special_token_ids] = -float('inf')

            best_token_id = token_logits.argmax(dim=-1)
            decompress_ids[best_position_to_fill[0], best_position_to_fill[1]] = best_token_id

        if return_ids_for_debug:
            reconstructed_text = self.tokenizer.decode(decompress_ids.squeeze(), skip_special_tokens=True)
            return reconstructed_text, decompress_ids.squeeze()

        return self.tokenizer.decode(decompress_ids.squeeze(), skip_special_tokens=True)

    def _calculate_compressed_size(self, compressed_data: Dict[str, any]) -> int:
        unmasked_tokens = compressed_data["unmasked_tokens"]
        unmasked_indices = compressed_data["unmasked_indices"]
        token_bits = len(unmasked_tokens) * 16
        index_bits = len(unmasked_indices) * 16
        metadata_bits = 16
        return token_bits + index_bits + metadata_bits

    def fine_tune(
        self,
        train_texts: List[str],
        eval_texts: List[str],
        epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 16,
        masking_probability: float = 0.5,
    ):
        learning_rate = float(learning_rate)
        from ..evaluation.metrics import CompressionMetrics

        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=128):
                self.tokenizer = tokenizer
                self.texts = texts
                self.max_length = max_length
            def __len__(self):
                return len(self.texts)
            def __getitem__(self, idx):
                return self.texts[idx]

        train_dataset = TextDataset(train_texts, self.tokenizer)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        num_training_steps = epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        self.model.train()

        metrics_calculator = CompressionMetrics(device=self.device)

        for epoch in range(epochs):
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_texts in progress_bar:
                optimizer.zero_grad()

                encodings = self.tokenizer(
                    batch_texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
                )
                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)

                labels = input_ids.clone()

                for i in range(input_ids.shape[0]):
                    mask_positions = self._get_random_mask_positions(input_ids[i], masking_probability)
                    input_ids[i, mask_positions] = self.mask_token_id
                    unmasked_positions = list(set(range(input_ids.shape[1])) - set(mask_positions))
                    labels[i, unmasked_positions] = -100

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                progress_bar.set_postfix(loss=loss.item())

            self.model.eval()
            total_word_acc, total_sem_sim, count = 0, 0, 0

            for text in eval_texts[:50]:
                if not text.strip():
                    continue
                try:
                    compressed_data = self.compress(text, masking_probability=masking_probability, strategy="random")
                    reconstructed_text = self.decompress(compressed_data)

                    total_word_acc += metrics_calculator.word_accuracy(text, reconstructed_text)
                    total_sem_sim += metrics_calculator.semantic_similarity(text, reconstructed_text)
                    count += 1
                except Exception as e:
                    continue

            avg_word_acc = total_word_acc / count if count > 0 else 0
            avg_sem_sim = total_sem_sim / count if count > 0 else 0

            print(f"Epoch {epoch+1} Evaluation - Word Accuracy: {avg_word_acc:.4f}, Semantic Similarity: {avg_sem_sim:.4f}")
            self.model.train()

    def calculate_model_size(self) -> dict[str, float]:
        param_count = sum(p.numel() for p in self.model.parameters())
        disk_size_mb = (param_count * 4) / (1024 ** 2)
        return {
            "disk_size_mb": disk_size_mb,
            "param_count":  param_count,
        }
