# src/models/predictive_masking.py
"""Predictive Masking compression implementation."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_compressor import BaseCompressor
import random


class PredictiveMaskingCompressor(BaseCompressor):
    """Compression via predictive masking of tokens."""

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def compress(
        self, text: str, masking_probability: float = 0.5, **kwargs
    ) -> Dict[str, any]:
        encoding = self.tokenizer(
            text, truncation=True, padding=False, return_tensors="pt", max_length=512
        )
        input_ids = encoding["input_ids"].squeeze() # FIX: Ensure 1D tensor

        # Determine which tokens are eligible for masking
        special_token_ids = self.tokenizer.all_special_ids
        eligible_indices = [
            i for i, token_id in enumerate(input_ids) if token_id not in special_token_ids
        ]
        
        # Determine number of tokens to mask
        num_to_mask = int(len(eligible_indices) * masking_probability)
        
        # Randomly select positions to mask from the eligible indices
        mask_positions = sorted(random.sample(eligible_indices, num_to_mask))
        
        # To calculate the compressed data, we only need to store the unmasked tokens and their positions.
        unmasked_indices = sorted(list(set(range(len(input_ids))) - set(mask_positions)))
        unmasked_token_ids = input_ids[unmasked_indices].cpu().numpy()

        return {
            "unmasked_tokens": unmasked_token_ids.astype(np.uint16),
            "unmasked_indices": np.array(unmasked_indices, dtype=np.uint16),
            "mask_positions": np.array(mask_positions, dtype=np.uint16),
            "original_length": len(input_ids),
            "masking_probability": masking_probability,
        }

    def decompress(self, compressed_data: Dict[str, any]) -> str:
        original_length = compressed_data["original_length"]
        unmasked_tokens = compressed_data["unmasked_tokens"]
        unmasked_indices = compressed_data["unmasked_indices"]
        
        # Reconstruct the initial masked sequence
        decompress_ids = torch.full((original_length,), self.mask_token_id, dtype=torch.long)
        decompress_ids[unmasked_indices] = torch.tensor(unmasked_tokens, dtype=torch.long)
        
        decompress_ids = decompress_ids.unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(decompress_ids)

        # Iteratively decompress
        for _ in range(2): # 2 passes can help resolve dependencies
            with torch.no_grad():
                outputs = self.model(input_ids=decompress_ids, attention_mask=attention_mask)
                predictions = outputs.logits
            
            # Find unresolved mask positions
            mask_indices = (decompress_ids == self.mask_token_id).nonzero(as_tuple=False).squeeze()
            if mask_indices.numel() == 0:
                break # No more masks to fill

            # Update predictions for masked positions
            predicted_ids = predictions[0, mask_indices].argmax(dim=-1)
            decompress_ids[0, mask_indices] = predicted_ids
        
        reconstructed_text = self.tokenizer.decode(
            decompress_ids.squeeze(), skip_special_tokens=True
        )

        return reconstructed_text

    def _calculate_compressed_size(self, compressed_data: Dict[str, any]) -> int:
        unmasked_tokens = compressed_data["unmasked_tokens"]
        unmasked_indices = compressed_data["unmasked_indices"]
        
        # Size of the unmasked tokens (assuming 16 bits per token for a vocab size up to 65536)
        token_bits = len(unmasked_tokens) * 16
        
        # Size of the indices (assuming 16 bits for sequence length up to 65536)
        index_bits = len(unmasked_indices) * 16
        
        # We also need to store the original length
        metadata_bits = 16 

        return token_bits + index_bits + metadata_bits

    def calculate_model_size(self) -> dict[str, float]:
        """
        Return a dictionary identical to the helpers in the other compressors.
        """
        param_count = sum(p.numel() for p in self.model.parameters())
        disk_size_mb = (param_count * 4) / (1024 ** 2)
        extras_mb = 0.0
        return {
            "disk_size_mb": disk_size_mb + extras_mb,
            "param_count":  param_count,
        }

    def adaptive_masking(self, text: str, target_ratio: float = 2.0) -> Dict[str, any]:
        low, high = 0.0, 1.0
        best_prob = 0.5
        best_ratio = 0.0

        for _ in range(10):
            mid = (low + high) / 2
            compressed = self.compress(text, masking_probability=mid)
            ratio = self.calculate_compression_ratio(text, compressed)
            if abs(ratio - target_ratio) < abs(best_ratio - target_ratio):
                best_prob = mid
                best_ratio = ratio
            if ratio < target_ratio:
                low = mid
            else:
                high = mid
        return self.compress(text, masking_probability=best_prob)
