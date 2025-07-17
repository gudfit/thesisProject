# src/experiments/finetune_experiment.py

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
                logger.info("Using cached fine-tuned model %s", cfg.model_id)
                continue
            tuner = ModelFineTuner(cfg.model_id, str(out_dir), self.config.training_args)
            tuner.fine_tune(self.config.dataset_name, self.config.dataset_subset)

    def _eval_domain(self, model, tok, sentences, domain: str, thresh: float, model_name: str, size_bytes: int, counts):
        rows = []
        sparsity = 1.0 - counts["nonzero_params"] / counts["total_params"] if counts["total_params"] else 0.0
        for s in tqdm(sentences, desc=f"{model_name} {domain}"):
            for theta in self.config.theta_budgets:
                lats = [self.reconstructor.reconstruct_sentence(model, tok, s, theta)[1] for _ in range(self.config.num_repetitions)]
                recon, _ = self.reconstructor.reconstruct_sentence(model, tok, s, theta)
                sim = self.metrics.calculate_semantic_similarity(s, recon)
                fact = self.metrics.calculate_factual_recall(s, recon)
                lex = self.metrics.lexical_recall(s, recon)
                succ_comp = 0.6*sim + 0.3*fact + 0.1*lex
                rows.append(dict(
                    model_name=model_name,
                    eval_domain=domain,
                    storage_cost_lambda=size_bytes,
                    storage_cost_bytes=size_bytes,
                    total_params=counts["total_params"],
                    nonzero_params=counts["nonzero_params"],
                    effective_param_bytes=counts["effective_param_bytes"],
                    sparsity=sparsity,
                    prompt_len_theta=theta,
                    retrieval_cost_ms=float(np.mean(lats)),
                    original_sentence=s,
                    reconstructed_sentence=recon,
                    is_perfect=self.metrics.is_perfect_match(s, recon),
                    semantic_similarity=sim,
                    factual_recall=fact,
                    lexical_recall=lex,
                    success_composite=succ_comp,
                    semantic_threshold_used=thresh,
                    is_semantically_equivalent=sim >= thresh,
                ))
        return rows

    def run_experiment(self) -> pd.DataFrame:
        id_sents = DataHandler.load_sentences(self.config.dataset_name, self.config.dataset_subset, self.config.test_split, max_samples=getattr(self.config,"max_samples",100))
        ood_sents = []
        if getattr(self.config,"ood_dataset_name",None):
            ood_sents = DataHandler.load_sentences(self.config.ood_dataset_name, self.config.ood_dataset_subset, self.config.ood_split, max_samples=getattr(self.config,"max_samples_ood",None) or getattr(self.config,"max_samples",100))
        rows = []
        for cfg in self.config.lambda_budgets:
            safe_name = cfg.model_id.replace("/", "_")
            model_dir = Path(self.config.finetune_output_dir) / safe_name
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, tok = ModelManager.load_model_and_tokenizer(str(model_dir), device)
            size_bytes = ModelManager.get_model_size_on_disk(str(model_dir))
            counts = ModelManager.count_nonzero_and_total_params(model)
            rows += self._eval_domain(model, tok, id_sents, "id", self.config.semantic_threshold, cfg.name, size_bytes, counts)
            if ood_sents:
                rows += self._eval_domain(model, tok, ood_sents, "ood", self.config.semantic_threshold_ood, cfg.name, size_bytes, counts)
            ModelManager.cleanup_model(model, tok)
        return pd.DataFrame(rows)

