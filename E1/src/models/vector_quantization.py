# src/models/vector_quantization.py
"""Vector Quantization compression implementation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_compressor import BaseCompressor
from sklearn.cluster import KMeans
from tqdm import tqdm
import joblib
import os

class VectorQuantizationCompressor(BaseCompressor):
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        self._load_model()
        self.codebook = None
        self.decoder = None
        self._init_decoder()

    def _init_decoder(self):
        from .latent_space_quantization import QuantizationDecoder
        config = self.model.config
        self.decoder = QuantizationDecoder(
            hidden_size=config.hidden_size, vocab_size=config.vocab_size
        ).to(self.device)

    def train_codebook(self, texts: List[str], num_clusters: int, model_path: str):
        self.codebook = KMeans(n_clusters=num_clusters, random_state=42, n_init=3, verbose=0)
        all_hidden_states = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), 16), desc=f"Generating vectors for VQ codebook (k={num_clusters})"):
                batch_texts = texts[i : i + 16]
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=128
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                outputs = (
                    self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                    if hasattr(self.model, "bert") else
                    self.model.roberta(input_ids=input_ids, attention_mask=attention_mask)
                    if hasattr(self.model, "roberta") else
                    self.model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
                )
                active_mask = attention_mask.bool()
                active_states = outputs.last_hidden_state[active_mask].cpu().numpy()
                all_hidden_states.append(active_states)

        all_hidden_states = np.vstack(all_hidden_states)
        self.codebook.fit(all_hidden_states)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.codebook, model_path)

    def load_codebook(self, path: str):
        self.codebook = joblib.load(path)

    def compress(self, text: str, **kwargs) -> Dict[str, any]:
        if self.codebook is None:
            raise RuntimeError("Codebook has not been trained or loaded.")

        encoding = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = (
                self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(self.model, "bert") else
                self.model.roberta(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(self.model, "roberta") else
                self.model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            )
            active_mask = attention_mask.bool().squeeze(0)
            hidden_states = outputs.last_hidden_state.squeeze(0)[active_mask].cpu().numpy()

        cluster_indices = self.codebook.predict(hidden_states)
        return {
            "cluster_indices": cluster_indices.astype(np.uint16),
            "original_length": len(cluster_indices),
            "num_clusters": self.codebook.n_clusters
        }

    def decompress(self, compressed_data: Dict[str, any]) -> str:
        if self.codebook is None:
            raise RuntimeError("Codebook has not been loaded.")

        cluster_indices = compressed_data["cluster_indices"]
        reconstructed_states = self.codebook.cluster_centers_[cluster_indices]
        reconstructed_states_tensor = torch.tensor(
            reconstructed_states, dtype=torch.float
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.decoder(reconstructed_states_tensor)
            predicted_ids = logits.argmax(dim=-1).squeeze()
            return self.tokenizer.decode(predicted_ids, skip_special_tokens=True)

    def _calculate_compressed_size(self, compressed_data: Dict[str, any]) -> int:
        num_indices = compressed_data["original_length"]
        num_clusters = compressed_data["num_clusters"]
        bits_per_index = int(np.ceil(np.log2(num_clusters)))
        return num_indices * bits_per_index

    def calculate_model_size(self) -> Dict[str, float]:
        param_count = (
            sum(p.numel() for p in self.model.parameters()) +
            sum(p.numel() for p in self.decoder.parameters())
        )
        disk_size_mb = (param_count * 4) / (1024 ** 2)
        codebook_mb = 0
        if self.codebook is not None:
            codebook_mb = (self.codebook.cluster_centers_.nbytes) / (1024 ** 2)
        return {
            "disk_size_mb": disk_size_mb + codebook_mb,
            "param_count": param_count,
            "codebook_size_mb": codebook_mb
        }

    def fine_tune(self, train_texts: List[str], eval_texts: List[str], **kwargs):
        num_clusters = kwargs.get("num_clusters")
        codebook_path = kwargs.get("codebook_path")

        if not num_clusters or not codebook_path:
            raise ValueError("fine_tune for VQ requires 'num_clusters' and 'codebook_path' in kwargs.")

        self.train_codebook(texts=train_texts, num_clusters=num_clusters, model_path=codebook_path)
        self.train_decoder(
            texts=train_texts,
            epochs=kwargs.get("epochs", 10),
            batch_size=kwargs.get("batch_size", 16),
            learning_rate=kwargs.get("learning_rate", 1e-3),
        )

    def train_decoder(self, texts: List[str], epochs: int = 10, batch_size: int = 16, learning_rate: float = 1e-3):
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        self.decoder.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            progress_bar = tqdm(range(0, len(texts), batch_size), desc=f"Training VQ Decoder Epoch {epoch+1}")
            for i in progress_bar:
                batch_texts = texts[i : i + batch_size]
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=128
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = (
                        self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(self.model, "bert") else
                        self.model.roberta(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(self.model, "roberta") else
                        self.model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
                    )
                    hidden_states = outputs.last_hidden_state

                logits = self.decoder(hidden_states)
                loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix(loss=total_loss/num_batches)
        self.decoder.eval()
