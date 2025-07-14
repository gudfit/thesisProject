import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
import logging
from typing import Dict, List, Any

from ..experiments.base_experiment import BaseExperiment
from ..models.advanced_pruning import AdvancedPruner, PruningConfig
from ..models.model_utils import ModelManager
from ..core.data_handler import DataHandler
from ..core.reconstruction import SentenceReconstructor
from ..core.metrics import MetricsCalculator
from ..models.finetuning import ModelFineTuner

logger = logging.getLogger(__name__)

class AdvancedPruningExperiment(BaseExperiment):
    """Experiment for testing advanced pruning methods."""
    
    def __init__(self, config):
        super().__init__(config)
        self.metrics_calculator = MetricsCalculator()
        self.reconstructor = SentenceReconstructor()
        
    def prepare_models(self):
        """Prepare pruned models using advanced methods."""
        for model_config in self.config.base_models:
            model_name_safe = model_config['model_id'].replace("/", "_")
            finetuned_path = Path(self.config.finetuned_models_dir) / model_name_safe
            
            # Ensure model is fine-tuned
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
            
            # Apply each pruning configuration
            for pruning_config in self.config.pruning_configs:
                self._apply_pruning_config(finetuned_path, model_config['name'], pruning_config)
    
    def _apply_pruning_config(self, model_path: Path, model_name: str, pruning_config_dict: dict):
        """Apply a specific pruning configuration."""
        # Extract configuration
        method = pruning_config_dict['method']
        amounts = pruning_config_dict['amounts']
        
        # Create pruning config
        config = PruningConfig(
            method=method,
            structured=pruning_config_dict.get('structured', False),
            block_size=pruning_config_dict.get('block_size', 4)
        )
        
        # Create output directory
        output_dir = Path(self.config.pruned_models_dir) / model_name / method
        
        # Create pruned models
        pruning_configs = [config]
        AdvancedPruner.create_pruned_model_suite(
            str(model_path),
            str(output_dir),
            pruning_configs,
            amounts
        )
    
    def run_experiment(self) -> pd.DataFrame:
        """Run the advanced pruning experiment."""
        logger.info("Running advanced pruning experiment...")
        
        # Load test sentences
        sentences = DataHandler.load_sentences(
            self.config.dataset_name,
            self.config.dataset_subset,
            self.config.test_split,
            max_samples=self.config.max_samples
        )
        
        all_results = []
        
        # Test each pruned model
        pruned_models_dir = Path(self.config.pruned_models_dir)
        for base_model_dir in pruned_models_dir.iterdir():
            if not base_model_dir.is_dir():
                continue
                
            base_model_name = base_model_dir.name
            
            for method_dir in base_model_dir.iterdir():
                if not method_dir.is_dir():
                    continue
                    
                method_name = method_dir.name
                
                # Load the base model architecture
                finetuned_path = Path(self.config.finetuned_models_dir) / base_model_name.replace("-", "_").lower()
                
                for pruned_file in method_dir.glob("*.pt"):
                    model_name = f"{base_model_name}_{pruned_file.stem}"
                    logger.info(f"Testing pruned model: {model_name}")
                    
                    # Load model architecture and weights
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model, tokenizer = ModelManager.load_model_and_tokenizer(str(finetuned_path), device)
                    
                    # Load pruned weights
                    model.load_state_dict(torch.load(pruned_file, map_location=device))
                    
                    # Get model stats
                    storage_cost = pruned_file.stat().st_size
                    nonzero_params = ModelManager.count_nonzero_params(model)
                    
                    # Extract pruning amount from filename
                    amount_str = pruned_file.stem.split('_')[-1]
                    if amount_str.isdigit():
                        pruning_amount = int(amount_str) / 100
                    else:
                        pruning_amount = 0.0
                    
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
                            
                            all_results.append({
                                'model_name': model_name,
                                'base_model': base_model_name,
                                'pruning_method': method_name,
                                'pruning_amount': pruning_amount,
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
