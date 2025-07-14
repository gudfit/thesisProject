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
    """Experiment for testing quantized models."""
    
    def __init__(self, config):
        super().__init__(config)
        self.metrics_calculator = MetricsCalculator()
        self.reconstructor = SentenceReconstructor()
        
    def prepare_models(self):
        """Prepare quantized models for the experiment."""
        # First ensure all models are fine-tuned
        for model_config in self.config.base_models:
            model_name_safe = model_config['model_id'].replace("/", "_")
            finetuned_path = Path(self.config.finetuned_models_dir) / model_name_safe
            
            if not finetuned_path.exists():
                logger.info(f"Fine-tuning {model_config['name']}...")
                finetuner = ModelFineTuner(
                    model_config['model_id'],
                    str(finetuned_path),
                    training_args={
                        'num_train_epochs': 1,
                        'per_device_train_batch_size': 4,
                        'logging_steps': 200
                    }
                )
                finetuner.fine_tune(self.config.dataset_name, self.config.dataset_subset)
            
            # Apply quantization configurations
            for quant_config in self.config.quantization_configs:
                self._apply_quantization(finetuned_path, model_config['name'], quant_config)
    
    def _apply_quantization(self, model_path: Path, model_name: str, quant_config: dict):
        """Apply quantization to a model."""
        config = QuantizationConfig(**quant_config)
        
        # Create output path
        quant_name = f"{model_name}_b{config.bits}_s{int(config.sparsity*100)}"
        output_path = Path(self.config.quantized_models_dir) / quant_name
        
        if output_path.exists():
            logger.info(f"Quantized model already exists: {quant_name}")
            return
        
        logger.info(f"Quantizing {model_name} with {config.bits}-bit, {config.sparsity*100}% sparsity")
        
        quantizer = SparseGPTQuantizer(config)
        quantizer.quantize_model(str(model_path), str(output_path))
        
    def run_experiment(self) -> pd.DataFrame:
        """Run the quantization experiment."""
        logger.info("Running quantization experiment...")
        
        # Load test sentences
        sentences = DataHandler.load_sentences(
            self.config.dataset_name,
            self.config.dataset_subset,
            self.config.test_split,
            max_samples=self.config.max_samples
        )
        
        all_results = []
        
        # Test each quantized model
        quantized_models_dir = Path(self.config.quantized_models_dir)
        for model_dir in quantized_models_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            logger.info(f"Testing quantized model: {model_name}")
            
            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, tokenizer = ModelManager.load_model_and_tokenizer(str(model_dir), device)
            
            # Get model stats
            storage_cost = ModelManager.get_model_size_on_disk(str(model_dir))
            nonzero_params = ModelManager.count_nonzero_params(model)
            
            # Test reconstruction
            for sentence in tqdm(sentences, desc=f"Testing {model_name}"):
                for theta in self.config.theta_budgets:
                    latencies = []
                    
                    for _ in range(self.config.num_repetitions):
                        recon_text, latency = self.reconstructor.reconstruct_sentence(
                            model, tokenizer, sentence, theta
                        )
                        latencies.append(latency)
                    
                    avg_latency = np.mean(latencies)
                    
                    # Calculate metrics
                    is_perfect = self.metrics_calculator.is_perfect_match(sentence, recon_text)
                    sem_sim = self.metrics_calculator.calculate_semantic_similarity(sentence, recon_text)
                    is_sem_eq = sem_sim >= self.config.semantic_threshold
                    
                    # Parse model configuration from name
                    parts = model_name.split('_')
                    base_model = '_'.join(parts[:-2])
                    bits = int(parts[-2][1:])  # Remove 'b' prefix
                    sparsity = int(parts[-1][1:]) / 100  # Remove 's' prefix and convert to decimal
                    
                    all_results.append({
                        'model_name': model_name,
                        'base_model': base_model,
                        'quantization_bits': bits,
                        'sparsity': sparsity,
                        'storage_cost_bytes': storage_cost,
                        'nonzero_params': nonzero_params,
                        'prompt_len_theta': theta,
                        'retrieval_cost_ms': avg_latency,
                        'original_sentence': sentence,
                        'reconstructed_sentence': recon_text,
                        'is_perfect': is_perfect,
                        'semantic_similarity': sem_sim,
                        'is_semantically_equivalent': is_sem_eq
                    })
            
            # Cleanup
            ModelManager.cleanup_model(model, tokenizer)
        
        return pd.DataFrame(all_results)
