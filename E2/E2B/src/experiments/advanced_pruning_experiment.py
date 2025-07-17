# src/experiments/advanced_pruning_experiment.py

import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
import logging
from typing import Dict, List

from .base_experiment import BaseExperiment
from ..models.advanced_pruning import AdvancedPruner, PruningConfig
from ..models.model_utils import ModelManager
from ..core.data_handler import DataHandler
from ..core.reconstruction import SentenceReconstructor
from ..core.metrics import MetricsCalculator
from ..models.finetuning import ModelFineTuner

logger = logging.getLogger(__name__)

class AdvancedPruningExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)
        self.metrics = MetricsCalculator()
        self.reconstructor = SentenceReconstructor()

    def prepare_models(self):
        for base in self.config.base_models:
            sid = base["model_id"].replace("/", "_")
            ft_root = Path(self.config.finetuned_models_dir)
            ft_path = ft_root / sid
            if not ft_path.exists():
                alt = ft_root / sid.lower()
                ft_path = alt if alt.exists() else ft_path
            if not ft_path.exists():
                logger.info("Fine-tuning %s", base["model_id"])
                ModelFineTuner(
                    base["model_id"],
                    str(ft_path),
                    training_args={"num_train_epochs":1,"per_device_train_batch_size":4,"logging_steps":200},
                ).fine_tune(self.config.dataset_name, self.config.dataset_subset)
            for p_cfg in self.config.pruning_configs:
                self._prune(ft_path, sid, p_cfg)

    def _prune(self, model_path: Path, sid: str, p_cfg: dict):
        cfg = PruningConfig(method=p_cfg["method"], structured=p_cfg.get("structured", False), block_size=p_cfg.get("block_size", 4))
        out_dir = Path(self.config.pruned_models_dir) / sid / p_cfg["method"]
        AdvancedPruner.create_pruned_model_suite(str(model_path), str(out_dir), [cfg], p_cfg["amounts"])

    def _eval_domain(self, model, tok, sents, domain, thresh, label, sid, meth, size, counts):
        rows = []
        sparsity = 1.0 - counts["nonzero_params"] / counts["total_params"] if counts["total_params"] else 0.0
        for sent in tqdm(sents, desc=f"{label} {domain}"):
            for theta in self.config.theta_budgets:
                lats = []
                for _ in range(self.config.num_repetitions):
                    _, lat = self.reconstructor.reconstruct_sentence(model, tok, sent, theta)
                    lats.append(lat)
                recon, _ = self.reconstructor.reconstruct_sentence(model, tok, sent, theta)
                sim = self.metrics.calculate_semantic_similarity(sent, recon)
                fact = self.metrics.calculate_factual_recall(sent, recon)
                lex = self.metrics.lexical_recall(sent, recon)
                succ_comp = 0.6*sim + 0.3*fact + 0.1*lex
                rows.append(dict(
                    model_name=label,
                    base_model=sid,
                    pruning_method=meth,
                    eval_domain=domain,
                    storage_cost_bytes=size,
                    total_params=counts["total_params"],
                    nonzero_params=counts["nonzero_params"],
                    effective_param_bytes=counts["effective_param_bytes"],
                    sparsity=sparsity,
                    prompt_len_theta=theta,
                    retrieval_cost_ms=float(np.mean(lats)),
                    original_sentence=sent,
                    reconstructed_sentence=recon,
                    is_perfect=self.metrics.is_perfect_match(sent, recon),
                    semantic_similarity=sim,
                    factual_recall=fact,
                    lexical_recall=lex,
                    success_composite=succ_comp,
                    semantic_threshold_used=thresh,
                    is_semantically_equivalent=sim >= thresh,
                ))
        return rows

    def run_experiment(self) -> pd.DataFrame:
        id_sents = DataHandler.load_sentences(self.config.dataset_name, self.config.dataset_subset, self.config.test_split, max_samples=self.config.max_samples)
        ood_sents = []
        if getattr(self.config,"ood_dataset_name",None):
            ood_sents = DataHandler.load_sentences(self.config.ood_dataset_name, self.config.ood_dataset_subset, self.config.ood_split, max_samples=getattr(self.config,"max_samples_ood",None) or self.config.max_samples)
        rows: List[Dict] = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pruned_root = Path(self.config.pruned_models_dir)
        for base_dir in pruned_root.iterdir():
            if not base_dir.is_dir():
                continue
            sid = base_dir.name
            ft_root = Path(self.config.finetuned_models_dir)
            ft_path = ft_root / sid
            if not ft_path.exists():
                alt = ft_root / sid.lower()
                ft_path = alt if alt.exists() else None
            if ft_path is None or not ft_path.exists():
                logger.warning("Skipping %s – no matching fine-tuned weights", sid)
                continue
            for meth_dir in base_dir.iterdir():
                if not meth_dir.is_dir():
                    continue
                for ckpt in meth_dir.glob("*.pt"):
                    label = f"{sid}_{ckpt.stem}"
                    model, tok = ModelManager.load_model_and_tokenizer(str(ft_path), device)
                    state = torch.load(ckpt, map_location=device)
                    model.load_state_dict(state)
                    counts = ModelManager.count_nonzero_and_total_params(model)
                    size = ckpt.stat().st_size
                    rows += self._eval_domain(model, tok, id_sents, "id", self.config.semantic_threshold, label, sid, meth_dir.name, size, counts)
                    if ood_sents:
                        rows += self._eval_domain(model, tok, ood_sents, "ood", self.config.semantic_threshold_ood, label, sid, meth_dir.name, size, counts)
                    ModelManager.cleanup_model(model, tok)
        return pd.DataFrame(rows)

