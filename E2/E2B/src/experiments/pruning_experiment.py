import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
import logging

from .base_experiment import BaseExperiment
from ..models.pruning import ModelPruner
from ..models.model_utils import ModelManager
from ..core.data_handler import DataHandler
from ..core.reconstruction import SentenceReconstructor
from ..core.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class PruningExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)
        self.metrics = MetricsCalculator()
        self.reconstructor = SentenceReconstructor()

    def prepare_models(self):
        finetuned_path = Path(self.config.finetuned_base_path)
        if not finetuned_path.exists():
            raise FileNotFoundError(f"Base model not found: {finetuned_path}")
        target_dir = Path(self.config.pruned_models_dir) / finetuned_path.name
        if not any(target_dir.glob("pruned_*.pt")):
            ModelPruner.create_pruned_models(
                base_model_path=str(finetuned_path),
                output_dir=str(target_dir),
                pruning_amounts=self.config.pruning_amounts,
            )

    def run_experiment(self) -> pd.DataFrame:
        sentences = DataHandler.load_sentences(
            self.config.dataset_name,
            self.config.dataset_subset,
            self.config.test_split,
            max_samples=getattr(self.config, "max_samples", 1000),
        )
        results = []
        pruned_dir = Path(self.config.pruned_models_dir) / Path(self.config.finetuned_base_path).name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for ckpt in pruned_dir.glob("pruned_*.pt"):
            label = f"{Path(self.config.base_model_id).name}_{ckpt.stem}"
            model, tok = ModelManager.load_model_and_tokenizer(self.config.finetuned_base_path, device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            storage = ckpt.stat().st_size
            for sent in tqdm(sentences, desc=f"Testing {label}"):
                for theta in self.config.theta_budgets:
                    latencies = []
                    for _ in range(self.config.num_repetitions):
                        recon, lat = self.reconstructor.reconstruct_sentence(model, tok, sent, theta)
                        latencies.append(lat)
                    recon, _ = self.reconstructor.reconstruct_sentence(model, tok, sent, theta)
                    sim = self.metrics.calculate_semantic_similarity(sent, recon)
                    results.append(
                        dict(
                            model_name=label,
                            storage_cost_bytes=storage,
                            prompt_len_theta=theta,
                            retrieval_cost_ms=float(np.mean(latencies)),
                            original_sentence=sent,
                            reconstructed_sentence=recon,
                            is_perfect=self.metrics.is_perfect_match(sent, recon),
                            semantic_similarity=sim,
                            is_semantically_equivalent=sim >= self.config.semantic_threshold,
                        )
                    )
            ModelManager.cleanup_model(model, tok)
        return pd.DataFrame(results)

