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
    """Fine‑tune the requested base models, generate pruned checkpoints, then
    benchmark them on sentence‑reconstruction."""

    def __init__(self, config):
        super().__init__(config)
        self.metrics = MetricsCalculator()
        self.reconstructor = SentenceReconstructor()

    # ------------------------------------------------------------------
    # Phase 1 – fine‑tune + prune
    # ------------------------------------------------------------------
    def prepare_models(self):
        for base in self.config.base_models:
            sid = base["model_id"].replace("/", "_")  # safe folder
            ft_root = Path(self.config.finetuned_models_dir)
            ft_path = ft_root / sid
            if not ft_path.exists():
                alt = ft_root / sid.lower()
                ft_path = alt if alt.exists() else ft_path
            if not ft_path.exists():
                logger.info("Fine‑tuning %s", base["model_id"])
                ModelFineTuner(
                    base["model_id"],
                    str(ft_path),
                    training_args={
                        "num_train_epochs": 1,
                        "per_device_train_batch_size": 4,
                        "logging_steps": 200,
                    },
                ).fine_tune(self.config.dataset_name, self.config.dataset_subset)
            for p_cfg in self.config.pruning_configs:
                self._prune(ft_path, sid, p_cfg)

    def _prune(self, model_path: Path, sid: str, p_cfg: dict):
        cfg = PruningConfig(
            method=p_cfg["method"],
            structured=p_cfg.get("structured", False),
            block_size=p_cfg.get("block_size", 4),
        )
        out_dir = Path(self.config.pruned_models_dir) / sid / p_cfg["method"]
        AdvancedPruner.create_pruned_model_suite(
            str(model_path), str(out_dir), [cfg], p_cfg["amounts"]
        )

    # ------------------------------------------------------------------
    # Phase 2 – evaluation
    # ------------------------------------------------------------------
    def run_experiment(self) -> pd.DataFrame:
        logger.info("Running advanced pruning experiment…")
        sents = DataHandler.load_sentences(
            self.config.dataset_name,
            self.config.dataset_subset,
            self.config.test_split,
            max_samples=self.config.max_samples,
        )
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
                logger.warning("Skipping %s – no matching fine‑tuned weights", sid)
                continue
            for meth_dir in base_dir.iterdir():
                if not meth_dir.is_dir():
                    continue
                for ckpt in meth_dir.glob("*.pt"):
                    label = f"{sid}_{ckpt.stem}"
                    logger.info("Testing %s", label)
                    model, tok = ModelManager.load_model_and_tokenizer(str(ft_path), device)
                    model.load_state_dict(torch.load(ckpt, map_location=device))
                    size = ckpt.stat().st_size
                    for sent in tqdm(sents, desc=label):
                        for theta in self.config.theta_budgets:
                            lats: List[float] = []
                            for _ in range(self.config.num_repetitions):
                                _, lat = self.reconstructor.reconstruct_sentence(model, tok, sent, theta)
                                lats.append(lat)
                            recon, _ = self.reconstructor.reconstruct_sentence(model, tok, sent, theta)
                            sim = self.metrics.calculate_semantic_similarity(sent, recon)
                            rows.append(
                                dict(
                                    model_name=label,
                                    base_model=sid,
                                    pruning_method=meth_dir.name,
                                    storage_cost_bytes=size,
                                    prompt_len_theta=theta,
                                    retrieval_cost_ms=float(np.mean(lats)),
                                    original_sentence=sent,
                                    reconstructed_sentence=recon,
                                    is_perfect=self.metrics.is_perfect_match(sent, recon),
                                    semantic_similarity=sim,
                                    is_semantically_equivalent=sim >= self.config.semantic_threshold,
                                )
                            )
                    ModelManager.cleanup_model(model, tok)
        return pd.DataFrame(rows)

