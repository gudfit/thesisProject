import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
import logging

from .base_experiment import BaseExperiment
from ..models.finetuning import ModelFineTuner
from ..models.model_utils import ModelManager
from ..core.data_handler import DataHandler
from ..core.reconstruction import SentenceReconstructor
from ..core.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class FinetuneExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)
        self.metrics = MetricsCalculator()
        self.reconstructor = SentenceReconstructor()

    def prepare_models(self):
        for cfg in self.config.lambda_budgets:
            safe_name = cfg.model_id.replace("/", "_")
            out_dir = Path(self.config.finetune_output_dir) / safe_name
            if out_dir.exists():
                logger.info("Using cached fine‑tuned model %s", cfg.model_id)
                continue
            tuner = ModelFineTuner(cfg.model_id, str(out_dir), self.config.training_args)
            tuner.fine_tune(self.config.dataset_name, self.config.dataset_subset)

    def run_experiment(self) -> pd.DataFrame:
        sentences = DataHandler.load_sentences(
            self.config.dataset_name,
            self.config.dataset_subset,
            self.config.test_split,
            max_samples=getattr(self.config, "max_samples", 100),
        )
        rows = []
        for cfg in self.config.lambda_budgets:
            safe_name = cfg.model_id.replace("/", "_")
            model_dir = Path(self.config.finetune_output_dir) / safe_name
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, tok = ModelManager.load_model_and_tokenizer(str(model_dir), device)
            size_bytes = ModelManager.get_model_size_on_disk(str(model_dir))
            for s in tqdm(sentences, desc=f"Testing {cfg.name}"):
                for theta in self.config.theta_budgets:
                    lats = [
                        self.reconstructor.reconstruct_sentence(model, tok, s, theta)[1]
                        for _ in range(self.config.num_repetitions)
                    ]
                    recon, _ = self.reconstructor.reconstruct_sentence(model, tok, s, theta)
                    sim = self.metrics.calculate_semantic_similarity(s, recon)
                    rows.append(
                        dict(
                            model_name=cfg.name,
                            storage_cost_lambda=size_bytes,
                            prompt_len_theta=theta,
                            retrieval_cost_ms=float(np.mean(lats)),
                            original_sentence=s,
                            reconstructed_sentence=recon,
                            is_perfect=self.metrics.is_perfect_match(s, recon),
                            semantic_similarity=sim,
                            is_semantically_equivalent=sim >= self.config.semantic_threshold,
                        )
                    )
            ModelManager.cleanup_model(model, tok)
        return pd.DataFrame(rows)

