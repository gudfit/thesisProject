# experiments/run_experiments.py
"""Main script to run all compression experiments."""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predictive_masking import PredictiveMaskingCompressor
from src.models.latent_space_quantization import LatentSpaceQuantizationCompressor
from src.models.vector_quantization import VectorQuantizationCompressor
from src.data.data_loader import WikiDataLoader
from src.evaluation.metrics import CompressionMetrics
from src.visualization.plots import CompressionVisualizer


class CompressionExperimentRunner:
    """Run compression experiments across models and parameters."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize experiment runner."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up logging
        self._setup_logging()

        # Set random seeds
        self._set_seeds(self.config['experiment']['seed'])

        # Initialize components
        self.data_loader = WikiDataLoader(
            self.config['data']['dataset_name'],
            self.config['data']['dataset_config']
        )
        self.metrics_calculator = CompressionMetrics(self.config['experiment']['device'])
        self.visualizer = CompressionVisualizer(
            style=self.config['visualization']['style'],
            figure_dir=os.path.join(self.config['experiment']['output_dir'], 'figures')
        )
        
        self.METRIC_KEYS = [
            'compression_ratio', 'word_accuracy', 'character_accuracy', 
            'semantic_similarity', 'rouge1_fmeasure', 'rouge2_fmeasure', 
            'rougeL_fmeasure', 'bert_score_f1', 'bits_per_character'
        ]

        # Create output directories
        self._create_output_dirs()

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = os.path.join(self.config['experiment']['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f'experiment_{datetime.now():%Y%m%d_%H%M%S}.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_output_dirs(self):
        """Create necessary output directories."""
        dirs = [
            self.config['experiment']['output_dir'],
            os.path.join(self.config['experiment']['output_dir'], 'figures'),
            os.path.join(self.config['experiment']['output_dir'], 'models'),
            os.path.join(self.config['experiment']['output_dir'], 'logs'),
            os.path.join(self.config['experiment']['output_dir'], 'results')
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def run_all_experiments(self):
        """Run experiments for all models and configurations."""
        self.logger.info("Starting compression experiments...")

        # Load data
        self.logger.info("Loading data...")
        train_texts, test_texts = self.data_loader.load_data(
            train_size=self.config['data']['train_size'],
            test_size=self.config['data']['test_size'],
            max_length=self.config['data']['max_length']
        )

        # Get diverse test samples
        test_samples = self.data_loader.get_diverse_test_samples(test_texts, num_samples=100)

        # Results storage
        all_results = {}

        # Run experiments for each model
        for model_config in self.config['models']:
            model_name = model_config['name']
            self.logger.info(f"\n======== Running experiments for {model_name} ========")

            # Run predictive masking experiments
            pm_results = self.run_predictive_masking_experiments(
                model_name, train_texts, test_samples
            )
            all_results[f"{model_name}_predictive_masking"] = pm_results

            # Run latent space quantization experiments
            lsq_results = self.run_latent_space_quantization_experiments(
                model_name, train_texts, test_samples
            )
            all_results[f"{model_name}_lsq"] = lsq_results
            
            # Run vector quantization experiments
            vq_results = self.run_vector_quantization_experiments(
                model_name, train_texts, test_samples
            )
            all_results[f"{model_name}_vq"] = vq_results

        # Save results
        self.save_results(all_results)

        # Create visualizations
        self.logger.info("Creating visualizations...")
        self.create_all_visualizations(all_results)

        self.logger.info("Experiments completed!")

    def run_predictive_masking_experiments(self, model_name: str, train_texts: List[str],
                                         test_samples: List[str]) -> Dict:
        """Run predictive masking experiments for a model."""
        self.logger.info(f"--- Starting Predictive Masking for {model_name} ---")
        results = {}

        compressor = PredictiveMaskingCompressor(
            model_name=model_name, device=self.config['experiment']['device']
        )
        
        # Fine-tune once
        if self.config['training']['epochs'] > 0:
            self.logger.info(f"Fine-tuning {model_name} for PM...")
            # --- THIS IS THE CORRECTED CALL ---
            compressor.fine_tune(
                train_texts=train_texts[:1000],
                eval_texts=test_samples, # Use test_samples for epoch validation
                epochs=self.config['training']['epochs'],
                learning_rate=self.config['training']['learning_rate'],
                batch_size=self.config['compression']['batch_size']
            )

        for masking_prob in self.config['compression']['masking_probabilities']:
            self.logger.info(f"Testing masking probability: {masking_prob}")
            prob_results = {key: [] for key in self.METRIC_KEYS}
            
            for text in tqdm(test_samples, desc=f"PM prob {masking_prob}", leave=False):
                try:
                    compressed = compressor.compress(text, masking_probability=masking_prob)
                    reconstructed = compressor.decompress(compressed)
                    metrics = self.metrics_calculator.calculate_all_metrics(text, reconstructed, compressed)
                    for key in self.METRIC_KEYS:
                        prob_results[key].append(metrics.get(key, 0))
                except Exception as e:
                    self.logger.error(f"Error processing text (PM): {text[:50]}... | {e}")
                    continue
            
            results[masking_prob] = {key: np.mean(values) if values else 0 for key, values in prob_results.items()}
        return results

    def run_latent_space_quantization_experiments(self, model_name: str, train_texts: List[str],
                                                test_samples: List[str]) -> Dict:
        """Run latent space quantization experiments for a model."""
        self.logger.info(f"--- Starting Latent Space Quantization for {model_name} ---")
        results = {}

        compressor = LatentSpaceQuantizationCompressor(
            model_name=model_name, device=self.config['experiment']['device']
        )
        
        # Train decoder once
        self.logger.info(f"Training LSQ decoder for {model_name}...")
        compressor.train_decoder(
            texts=train_texts[:1000], epochs=5, batch_size=self.config['compression']['batch_size']
        )

        for bits in self.config['compression']['quantization_bits_levels']:
            self.logger.info(f"Testing {bits}-bit LSQ")
            prob_results = {key: [] for key in self.METRIC_KEYS}

            for text in tqdm(test_samples, desc=f"LSQ {bits}-bit", leave=False):
                try:
                    compressed = compressor.compress(text, quantization_bits=bits)
                    reconstructed = compressor.decompress(compressed)
                    metrics = self.metrics_calculator.calculate_all_metrics(text, reconstructed, compressed)
                    for key in self.METRIC_KEYS:
                        prob_results[key].append(metrics.get(key, 0))
                except Exception as e:
                    self.logger.error(f"Error processing text (LSQ): {text[:50]}... | {e}")
                    continue
            
            results[bits] = {key: np.mean(values) if values else 0 for key, values in prob_results.items()}
            results[bits]['quantization_bits'] = bits
        return results
        
    def run_vector_quantization_experiments(self, model_name: str, train_texts: List[str],
                                            test_samples: List[str]) -> Dict:
        """Run vector quantization experiments for a model."""
        self.logger.info(f"--- Starting Vector Quantization for {model_name} ---")
        results = {}

        compressor = VectorQuantizationCompressor(
            model_name=model_name, device=self.config['experiment']['device']
        )
        
        # Train decoder once
        self.logger.info(f"Training VQ decoder for {model_name}...")
        compressor.train_decoder(
            texts=train_texts[:1000], epochs=5, batch_size=self.config['compression']['batch_size']
        )

        for k in self.config['compression']['vq_codebook_sizes']:
            self.logger.info(f"Testing VQ with {k} clusters...")
            
            # Train the codebook for this k
            codebook_path = os.path.join(
                self.config['experiment']['output_dir'], 'models', f"{model_name.replace('/', '_')}_vq_codebook_k{k}.joblib"
            )
            compressor.train_codebook(train_texts[:2000], num_clusters=k, model_path=codebook_path)
            
            prob_results = {key: [] for key in self.METRIC_KEYS}

            for text in tqdm(test_samples, desc=f"VQ k={k}", leave=False):
                try:
                    compressed = compressor.compress(text)
                    reconstructed = compressor.decompress(compressed)
                    metrics = self.metrics_calculator.calculate_all_metrics(text, reconstructed, compressed)
                    for key in self.METRIC_KEYS:
                        prob_results[key].append(metrics.get(key, 0))
                except Exception as e:
                    self.logger.error(f"Error processing text (VQ): {text[:50]}... | {e}")
                    continue
            
            results[k] = {key: np.mean(values) if values else 0 for key, values in prob_results.items()}
            results[k]['codebook_size'] = k
        return results

    def save_results(self, results: Dict):
        """Save experimental results."""
        results_path = os.path.join(
            self.config['experiment']['output_dir'], 'results', f'results_{datetime.now():%Y%m%d_%H%M%S}.json'
        )
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.generic): return obj.item()
            if isinstance(obj, dict): return {key: convert_to_serializable(value) for key, value in obj.items()}
            if isinstance(obj, list): return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_results = convert_to_serializable(results)
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        self.logger.info(f"Results saved to {results_path}")

        csv_data = []
        for model_method, model_results in results.items():
            for param, metrics in model_results.items():
                if isinstance(param, (float, int)):
                    row = {'model_method': model_method, 'hyperparameter': param,
                           **{k: v for k, v in metrics.items() if not isinstance(v, list)}}
                    csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = results_path.replace('.json', '.csv')
            df.to_csv(csv_path, index=False)
            self.logger.info(f"CSV results saved to {csv_path}")

    def create_all_visualizations(self, results: Dict):
        """Create all visualizations from results."""
        self.visualizer.plot_efficiency_vs_fidelity_tradeoff(
            results, save_name="all_methods_tradeoff"
        )
        
        pm_results = {k: v for k, v in results.items() if 'predictive_masking' in k}
        lsq_results = {k: v for k, v in results.items() if 'lsq' in k}
        vq_results = {k: v for k, v in results.items() if 'vq' in k}

        if pm_results: self.visualizer.create_summary_report(pm_results, "predictive_masking")
        if lsq_results: self.visualizer.create_summary_report(lsq_results, "latent_space_quantization")
        if vq_results: self.visualizer.create_summary_report(vq_results, "vector_quantization")

        self.logger.info("All visualizations created!")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM compression experiments")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    runner = CompressionExperimentRunner(args.config)
    runner.run_all_experiments()

if __name__ == "__main__":
    main()
