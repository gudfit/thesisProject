# src/models/simple_lossy_compressor.py
"""A simple, non-neural lossy compressor for baseline comparison."""

from typing import Dict, List

class SimpleLossyCompressor:
    def compress(self, text: str, n: int = 10) -> str:
        if n <= 0:
            return ""
        words = text.split()
        reconstructed_words = [word for i, word in enumerate(words) if (i + 1) % n != 0]
        return " ".join(reconstructed_words)

    def get_bits_per_character(self, original_text: str, compressed_text: str) -> float:
        original_bits = len(original_text.encode('utf-8', 'ignore')) * 8
        compressed_bits = len(compressed_text.encode('utf-8', 'ignore')) * 8
        metadata_bits = 32
        total_compressed_bits = compressed_bits + metadata_bits
        return total_compressed_bits / len(original_text) if original_text else 0
