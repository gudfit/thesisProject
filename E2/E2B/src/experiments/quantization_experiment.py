# src/experiments/quantization_experiment.py

import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
import logging
from typing import Dict, List, Any

from ..experiments.base_experiment import BaseExperiment
from ..models.quantization import SparseGPTQuantizer, QuantizationConfig
from ..models.model_utils import ModelManager
from ..core.data_handler import DataHandler
from ..core.reconstruction import SentenceReconstructor
from ..core.metrics import MetricsCalculator
from ..models.finetuning import ModelFineTuner

logger = logging.getLogger(__name__)

class QuantizationExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)
        self.metrics_calculator = MetricsCalculator()
        self.reconstructor = SentenceReconstructor()

    def prepare_models(self):
        for model_config in self.config.base_models:
            model_name_safe = model_config['model_id'].replace("/", "_")
            finetuned_path = Path(self.config.finetuned_models_dir) / model_name_safe
            if not finetuned_path.exists():
                logger.info(f"Fine-tuning {model_config['name']}...")
                finetuner = ModelFineTuner(
                    model_config['model_id'],
                    str(finetuned_path),
                    training_args={'num_train_epochs':1,'per_device_train_batch_size':4,'logging_steps':200}
                )
                finetuner.fine_tune(self.config.dataset_name, self.config.dataset_subset)
            for quant_config in self.config.quantization_configs:
                self._apply_quantization(finetuned_path, model_config['name'], quant_config)

    def _apply_quantization(self, model_path: Path, model_name: str, quant_config: dict):
        config = QuantizationConfig(**quant_config)
        quant_name = f"{model_name}_b{config.bits}_s{int(config.sparsity*100)}"
        output_path = Path(self.config.quantized_models_dir) / quant_name
        if output_path.exists():
            logger.info(f"Quantized model already exists: {quant_name}")
            return
        logger.info(f"Quantizing {model_name} with {config.bits}-bit, {config.sparsity*100}% sparsity")
        quantizer = SparseGPTQuantizer(config)
        quantizer.quantize_model(str(model_path), str(output_path))

    def _eval_domain(self, model, tokenizer, sentences, domain, thresh, model_name, storage_cost, total_params, nonzero_params, eff_bytes, sparsity, base_model, bits, sparsity_name):
        rows = []
        for sentence in tqdm(sentences, desc=f"{model_name} {domain}"):
            for theta in self.config.theta_budgets:
                latencies = []
                for _ in range(self.config.num_repetitions):
                    recon_text, latency = self.reconstructor.reconstruct_sentence(model, tokenizer, sentence, theta)
                    latencies.append(latency)
                avg_latency = np.mean(latencies)
                recon_text, _ = self.reconstructor.reconstruct_sentence(model, tokenizer, sentence, theta)
                is_perfect = self.metrics_calculator.is_perfect_match(sentence, recon_text)
                sem_sim = self.metrics_calculator.calculate_semantic_similarity(sentence, recon_text)
                fact = self.metrics_calculator.calculate_factual_recall(sentence, recon_text)
                lex = self.metrics_calculator.lexical_recall(sentence, recon_text)
                succ_comp = 0.6*sem_sim + 0.3*fact + 0.1*lex
                is_sem_eq = sem_sim >= thresh
                rows.append({
                    'model_name': model_name,
                    'base_model': base_model,
                    'quantization_bits': bits,
                    'sparsity': sparsity_name if sparsity_name is not None else sparsity,
                    'eval_domain': domain,
                    'storage_cost_bytes': storage_cost,
                    'total_params': total_params,
                    'nonzero_params': nonzero_params,
                    'effective_param_bytes': eff_bytes,
                    'prompt_len_theta': theta,
                    'retrieval_cost_ms': avg_latency,
                    'original_sentence': sentence,
                    'reconstructed_sentence': recon_text,
                    'is_perfect': is_perfect,
                    'semantic_similarity': sem_sim,
                    'factual_recall': fact,
                    'lexical_recall': lex,
                    'success_composite': succ_comp,
                    'semantic_threshold_used': thresh,
                    'is_semantically_equivalent': is_sem_eq
                })
        return rows

    def run_experiment(self) -> pd.DataFrame:
        logger.info("Running quantization experiment...")
        id_sents = DataHandler.load_sentences(self.config.dataset_name, self.config.dataset_subset, self.config.test_split, max_samples=self.config.max_samples)
        ood_sents = []
        if getattr(self.config,"ood_dataset_name",None):
            ood_sents = DataHandler.load_sentences(self.config.ood_dataset_name, self.config.ood_dataset_subset, self.config.ood_split, max_samples=getattr(self.config,"max_samples_ood",None) or self.config.max_samples)
        all_results = []
        quantized_models_dir = Path(self.config.quantized_models_dir)
        for model_dir in quantized_models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            logger.info(f"Testing quantized model: {model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, tokenizer = ModelManager.load_model_and_tokenizer(str(model_dir), device)
            storage_cost = ModelManager.get_model_size_on_disk(str(model_dir))
            counts = ModelManager.count_nonzero_and_total_params(model)
            nonzero_params = counts["nonzero_params"]
            total_params = counts["total_params"]
            eff_bytes = counts["effective_param_bytes"]
            sparsity = 1.0 - nonzero_params / total_params if total_params else 0.0
            parts = model_name.split('_')
            base_model = '_'.join(parts[:-2]) if len(parts) >= 3 else model_name
            try:
                bits = int(parts[-2][1:]) if len(parts) >= 2 and parts[-2].startswith('b') else None
            except Exception:
                bits = None
            try:
                sparsity_name = int(parts[-1][1:]) / 100 if len(parts) >= 1 and parts[-1].startswith('s') else None
            except Exception:
                sparsity_name = None
            all_results += self._eval_domain(model, tokenizer, id_sents, "id", self.config.semantic_threshold, model_name, storage_cost, total_params, nonzero_params, eff_bytes, sparsity, base_model, bits, sparsity_name)
            if ood_sents:
                all_results += self._eval_domain(model, tokenizer, ood_sents, "ood", self.config.semantic_threshold_ood, model_name, storage_cost, total_params, nonzero_params, eff_bytes, sparsity, base_model, bits, sparsity_name)
            ModelManager.cleanup_model(model, tokenizer)
        return pd.DataFrame(all_results)

