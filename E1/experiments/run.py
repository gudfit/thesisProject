
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predictive_masking import PredictiveMaskingCompressor
from src.models.latent_space_quantization import LatentSpaceQuantizationCompressor
from src.models.vector_quantization import VectorQuantizationCompressor
from src.data.data_loader import WikiDataLoader
from src.evaluation.metrics import CompressionMetrics
from src.visualization.plots import CompressionVisualizer


class CompressionExperimentRunner:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._create_output_dirs()
        self._setup_logging()
        self._set_seeds(self.config.get('experiment', {}).get('seed', 42))

        self.data_loader = WikiDataLoader(
            self.config.get('data', {}).get('dataset_name', 'wikitext'),
            self.config.get('data', {}).get('dataset_config', 'wikitext-2-v1')
        )
        self.metrics_calculator = CompressionMetrics(self.config.get('experiment', {}).get('device', 'cuda'))
        self.visualizer = CompressionVisualizer(
            figure_dir=os.path.join(self.config.get('experiment', {}).get('output_dir', './results'), 'figures')
        )

        self.METRIC_KEYS = [
            'compression_ratio', 'word_accuracy', 'character_accuracy',
            'semantic_similarity', 'rouge1_fmeasure', 'rouge2_fmeasure',
            'rougeL_fmeasure', 'bert_score_f1', 'bits_per_character'
        ]

    def _setup_logging(self):
        log_dir = os.path.join(self.config.get('experiment', {}).get('output_dir', './results'), 'logs')
        log_file = os.path.join(log_dir, f'experiment_{datetime.now():%Y%m%d_%H%M%S}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("absl").setLevel(logging.ERROR)
        self.logger = logging.getLogger(__name__)

    def _set_seeds(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_output_dirs(self):
        output_dir = self.config.get('experiment', {}).get('output_dir', './results')
        dirs = [
            output_dir,
            os.path.join(output_dir, 'figures'),
            os.path.join(output_dir, 'models'),
            os.path.join(output_dir, 'logs'),
            os.path.join(output_dir, 'results')
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def run_all_experiments(self):
        self.logger.info("Starting all compression experiments.")

        data_conf = self.config.get('data', {})
        exp_conf = self.config.get('experiment', {})

        self.logger.info("Loading and preparing data...")
        train_texts, test_texts = self.data_loader.load_data(
            train_size=data_conf.get('train_size', 10000),
            test_size=data_conf.get('test_size', 1000),
            max_length=data_conf.get('max_length', 256)
        )

        test_samples = self.data_loader.get_diverse_test_samples(
            test_texts, num_samples=exp_conf.get('num_test_samples', 100)
        )

        all_results = {}
        for model_config in self.config.get('models', []):
            model_name = model_config['name']
            self.logger.info(f"\n{'='*20} Processing Model: {model_name} {'='*20}")

            all_results[f"{model_name}_predictive_masking"] = self.run_predictive_masking_experiments(model_name, train_texts, test_samples)
            all_results[f"{model_name}_lsq"] = self.run_latent_space_quantization_experiments(model_name, train_texts, test_samples)
            all_results[f"{model_name}_vq"] = self.run_vector_quantization_experiments(model_name, train_texts, test_samples)

        self.logger.info("Saving all results...")
        self.save_results(all_results)

        self.logger.info("Creating visualizations...")
        self.create_all_visualizations(all_results)

        self.logger.info("All experiments completed successfully.")

    def run_predictive_masking_experiments(self, model_name: str, train_texts: List[str], test_samples: List[str]) -> Dict:
        self.logger.info(f"--- Running Predictive Masking for {model_name} ---")
        results = {}
        compressor = PredictiveMaskingCompressor(model_name=model_name, device=self.config.get('experiment', {}).get('device', 'cuda'))

        training_conf = self.config.get('training', {})
        data_conf = self.config.get('data', {})

        if training_conf.get('epochs', 0) > 0:
            self.logger.info(f"Fine-tuning {model_name} for Predictive Masking...")
            compressor.fine_tune(
                train_texts=train_texts[:data_conf.get('fine_tune_size', 1000)],
                eval_texts=test_samples,
                epochs=training_conf.get('epochs', 3),
                learning_rate=training_conf.get('learning_rate', 5e-5),
                batch_size=self.config.get('compression', {}).get('batch_size', 16),
                masking_probability=training_conf.get('training_masking_probability', 0.5)
            )

        for masking_prob in self.config.get('compression', {}).get('masking_probabilities', [0.5]):
            results[masking_prob] = self._evaluate_compressor(
                compressor, test_samples, f"PM prob {masking_prob}",
                masking_probability=masking_prob, strategy='random'
            )
        return results

    def run_latent_space_quantization_experiments(self, model_name: str, train_texts: List[str], test_samples: List[str]) -> Dict:
        self.logger.info(f"--- Running Latent Space Quantization for {model_name} ---")
        results = {}
        compressor = LatentSpaceQuantizationCompressor(model_name=model_name, device=self.config.get('experiment', {}).get('device', 'cuda'))

        training_conf = self.config.get('training', {})
        data_conf = self.config.get('data', {})

        self.logger.info(f"Training LSQ decoder for {model_name}...")
        compressor.fine_tune(
            train_texts=train_texts[:data_conf.get('fine_tune_size', 1000)],
            eval_texts=test_samples,
            epochs=training_conf.get('epochs', 5),
            learning_rate=training_conf.get('learning_rate', 1e-3),
            batch_size=self.config.get('compression', {}).get('batch_size', 16)
        )

        for bits in self.config.get('compression', {}).get('quantization_bits_levels', [8]):
            results[bits] = self._evaluate_compressor(
                compressor, test_samples, f"LSQ {bits}-bit", quantization_bits=bits
            )
        return results

    def run_vector_quantization_experiments(self, model_name: str, train_texts: List[str], test_samples: List[str]) -> Dict:
        self.logger.info(f"--- Running Vector Quantization for {model_name} ---")
        results = {}
        compressor = VectorQuantizationCompressor(model_name=model_name, device=self.config.get('experiment', {}).get('device', 'cuda'))

        output_dir = self.config.get('experiment', {}).get('output_dir', './results')
        models_dir = os.path.join(output_dir, 'models')
        training_conf = self.config.get('training', {})
        data_conf = self.config.get('data', {})

        for k in self.config.get('compression', {}).get('vq_codebook_sizes', [256]):
            self.logger.info(f"Training VQ components for k={k}...")
            codebook_path = os.path.join(models_dir, f"{model_name.replace('/', '_')}_vq_k{k}.joblib")

            compressor.fine_tune(
                train_texts=train_texts[:data_conf.get('fine_tune_size', 1000)],
                eval_texts=test_samples,
                epochs=training_conf.get('epochs', 5),
                learning_rate=training_conf.get('learning_rate', 1e-3),
                batch_size=self.config.get('compression', {}).get('batch_size', 16),
                num_clusters=k,
                codebook_path=codebook_path
            )

            results[k] = self._evaluate_compressor(compressor, test_samples, f"VQ k={k}")
        return results

    def _evaluate_compressor(self, compressor, test_samples: List[str], desc: str, **kwargs) -> Dict:
        self.logger.info(f"Evaluating: {desc}...")
        prob_results = {key: [] for key in self.METRIC_KEYS}

        for text in tqdm(test_samples, desc=desc, leave=False):
            if not text.strip(): continue
            try:
                compressed = compressor.compress(text, **kwargs)
                reconstructed = compressor.decompress(compressed)
                metrics = self.metrics_calculator.calculate_all_metrics(text, reconstructed, compressed, compressor)
                for key in self.METRIC_KEYS:
                    prob_results[key].append(metrics.get(key, 0))
            except Exception as e:
                self.logger.error(f"Error processing text with {desc}: {text[:50]}... | {e}", exc_info=False)

        return {key: np.mean(values) if values else 0 for key, values in prob_results.items()}

    def save_results(self, results: Dict):
        results_dir = os.path.join(self.config.get('experiment', {}).get('output_dir', './results'), 'results')
        results_path = os.path.join(results_dir, f'results_{datetime.now():%Y%m%d_%H%M%S}.json')

        def convert_to_serializable(obj):
            if isinstance(obj, np.generic): return obj.item()
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj

        serializable_results = {k: {p: {m: convert_to_serializable(v) for m, v in metrics.items()} for p, metrics in params.items()} for k, params in results.items()}

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        self.logger.info(f"JSON results saved to {results_path}")

        csv_data = []
        for model_method, model_results in results.items():
            for param, metrics in model_results.items():
                row = {'model_method': model_method, 'hyperparameter': param, **metrics}
                csv_data.append(row)

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = results_path.replace('.json', '.csv')
            df.to_csv(csv_path, index=False)
            self.logger.info(f"CSV results saved to {csv_path}")

    def create_all_visualizations(self, results: Dict):
        self.visualizer.plot_efficiency_vs_fidelity_tradeoff(results, save_name="all_methods_tradeoff")

        grouped_results = {}
        for key, value in results.items():
            if 'predictive_masking' in key:
                method_type = 'predictive_masking'
            elif 'lsq' in key:
                method_type = 'lsq'
            elif 'vq' in key:
                method_type = 'vq'
            else:
                continue

            if method_type not in grouped_results: grouped_results[method_type] = {}
            grouped_results[method_type][key] = value

        if 'predictive_masking' in grouped_results: self.visualizer.create_summary_report(grouped_results['predictive_masking'], "predictive_masking")
        if 'lsq' in grouped_results: self.visualizer.create_summary_report(grouped_results['lsq'], "latent_space_quantization")
        if 'vq' in grouped_results: self.visualizer.create_summary_report(grouped_results['vq'], "vector_quantization")

        self.logger.info("All visualizations created!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM compression experiments")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    runner = CompressionExperimentRunner(config_path=args.config)
    runner.run_all_experiments()

if __name__ == "__main__":
    main()
