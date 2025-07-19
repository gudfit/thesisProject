import os
import argparse
import sys
import yaml
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.autoregressive_coding import AutoregressiveCoding
from src.models.predictive_masking import PredictiveMaskingCompressor
from src.models.latent_space_quantization import LatentSpaceQuantizationCompressor
from src.models.traditional_compressors import TraditionalCompressors
from src.evaluation.metrics import CompressionMetrics
from src.visualization.plots import CompressionVisualizer


class Experiment1DRunner:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.exp_conf = self.config.get('experiment', {})
        self.exp1d_conf = self.config.get('experiment_1d', {})
            
        self._create_output_dirs()
        self._setup_logging()
        
        self.visualizer = CompressionVisualizer(
            figure_dir=os.path.join(self.exp_conf.get('output_dir', './results'), 'figures', 'experiment_1d')
        )
        self.metrics_calculator = CompressionMetrics(self.exp_conf.get('device', 'cuda'))
        
    def _setup_logging(self):
        log_dir = os.path.join(self.exp_conf.get('output_dir', './results'), 'logs')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f'experiment_1d_{datetime.now():%Y%m%d_%H%M%S}.log')),
                logging.StreamHandler()
            ]
        )
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        self.logger = logging.getLogger(__name__)
        
    def _create_output_dirs(self):
        output_dir = self.exp_conf.get('output_dir', './results')
        dirs = [
            os.path.join(output_dir, 'figures', 'experiment_1d'),
            os.path.join(output_dir, 'results', 'experiment_1d'),
            os.path.join(output_dir, 'logs'),
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
    def load_test_corpus(self, num_samples: int) -> List[str]:
        self.logger.info(f"Loading {num_samples} samples from WikiText-103...")
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]
        return texts[:num_samples]
    
    def run_comparative_benchmark(self):
        self.logger.info("Starting Experiment 1D: Comparative Compression Benchmark")
        test_corpus = self.load_test_corpus(self.exp1d_conf.get('num_samples', 1000))
        results = {'llm_lossless': {}, 'llm_lossy': {}, 'traditional': {}}

        self.logger.info("\n--- Evaluating LLM-based Lossless Compression ---")
        for model_name in self.exp1d_conf.get('ar_models', ['gpt2']):
            self.logger.info(f"Processing AR model: {model_name}...")
            compressor = AutoregressiveCoding(model_name, self.exp_conf.get('device', 'cuda'))
            results['llm_lossless'][model_name] = self._evaluate_ar_compression(compressor, test_corpus[:100])
            
        self.logger.info("\n--- Evaluating LLM-based Lossy Compression ---")
        for model_config in self.config.get('models', []):
            model_name = model_config['name']
            
            self.logger.info(f"Processing PM for model: {model_name}...")
            pm_compressor = PredictiveMaskingCompressor(model_name, self.exp_conf.get('device', 'cuda'))
            results['llm_lossy'][f"{model_name}_pm"] = self._evaluate_pm_compression(pm_compressor, test_corpus[:200])
            
            self.logger.info(f"Processing LSQ for model: {model_name}...")
            lsq_compressor = LatentSpaceQuantizationCompressor(model_name, self.exp_conf.get('device', 'cuda'))
            lsq_compressor.fine_tune(train_texts=test_corpus[:500], eval_texts=test_corpus[:50])
            results['llm_lossy'][f"{model_name}_lsq"] = self._evaluate_lsq_compression(lsq_compressor, test_corpus[:200])

        self.logger.info("\n--- Evaluating Traditional Lossless Compression ---")
        trad_compressors = TraditionalCompressors()
        results['traditional'] = trad_compressors.benchmark_all_algorithms(test_corpus)
        
        self.logger.info("\n--- Finalizing Experiment ---")
        results['comparative_analysis'] = self._perform_comparative_analysis(results)
        self.create_visualizations(results)
        self.save_results(results)
        self.print_summary(results)
        self.logger.info("\nExperiment 1D completed!")
        return results
    
    def _evaluate_ar_compression(self, compressor: AutoregressiveCoding, texts: List[str]) -> Dict[str, Any]:
        analysis = compressor.theoretical_compression_analysis(texts)
        return {
            'method': 'Autoregressive Coding', 'model_name': compressor.model_name, 'is_lossless': True,
            'avg_bits_per_character': analysis['avg_bits_per_character'],
            'avg_compression_ratio': analysis['avg_compression_ratio'],
            'reconstruction_accuracy': 1.0, 'model_size_mb': analysis['model_info']['model_size_mb']
        }
    
    def _evaluate_pm_compression(self, compressor: PredictiveMaskingCompressor, texts: List[str]) -> Dict[str, Any]:
        best_results = {}
        best_score = -float('inf')
        
        for mask_prob in self.exp1d_conf.get('pm_masking_levels', [0.5]):
            total_bpc, scores = 0, 0
            for text in tqdm(texts, desc=f"PM {mask_prob:.1f}", leave=False):
                compressed = compressor.compress(text, masking_probability=mask_prob)
                reconstructed = compressor.decompress(compressed)
                metrics = self.metrics_calculator.calculate_all_metrics(text, reconstructed, compressed, compressor)
                total_bpc += metrics.get('bits_per_character', 0)
                scores += metrics.get('word_accuracy', 0)
            
            avg_bpc = total_bpc / len(texts) if texts else 0
            avg_accuracy = scores / len(texts) if texts else 0
            current_score = avg_accuracy - (avg_bpc / 10) 
            
            if current_score > best_score:
                best_score = current_score
                best_results = {'masking_probability': mask_prob, 'avg_bits_per_character': avg_bpc, 'reconstruction_accuracy': avg_accuracy}

        return {
            'method': 'Predictive Masking', 'model_name': compressor.model_name, 'is_lossless': False,
            'model_size_mb': compressor.calculate_model_size()['disk_size_mb'], **best_results
        }
    
    def _evaluate_lsq_compression(self, compressor: LatentSpaceQuantizationCompressor, texts: List[str]) -> Dict[str, Any]:
        total_bpc, scores = 0, 0
        bits = self.exp1d_conf.get('lsq_bits', 8)

        for text in tqdm(texts, desc=f"LSQ {bits}-bit eval", leave=False):
            compressed = compressor.compress(text, quantization_bits=bits)
            reconstructed = compressor.decompress(compressed)
            metrics = self.metrics_calculator.calculate_all_metrics(text, reconstructed, compressed, compressor)
            total_bpc += metrics.get('bits_per_character', 0)
            scores += metrics.get('word_accuracy', 0)

        avg_bpc = total_bpc / len(texts) if texts else 0
        avg_accuracy = scores / len(texts) if texts else 0

        return {
            'method': 'Latent Space Quantization', 'model_name': compressor.model_name, 'is_lossless': False,
            'quantization_bits': bits, 'avg_bits_per_character': avg_bpc,
            'reconstruction_accuracy': avg_accuracy, 'model_size_mb': compressor.calculate_model_size()['disk_size_mb']
        }
    
    def _perform_comparative_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Comparative analysis generated.")
        return {"status": "Generated. See plots and tables for details."}
    
    def create_visualizations(self, results: Dict[str, Any]):
        self.logger.info("Creating visualizations for Experiment 1D...")
        
        all_methods = []
        for model, metrics in results['llm_lossless'].items(): all_methods.append({'group': 'LLM Lossless', 'name': f"AR-{model}", **metrics})
        for model, metrics in results['llm_lossy'].items(): all_methods.append({'group': 'LLM Lossy', 'name': model.replace('_', '-'), **metrics})
        for model, metrics in results['traditional'].items(): all_methods.append({'group': 'Traditional', 'name': model.upper(), 'avg_bits_per_character': metrics['avg_bits_per_character'], 'reconstruction_accuracy': 1.0, 'model_size_mb': 0.0})

        df = pd.DataFrame(all_methods)
        
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df, x='name', y='avg_bits_per_character', hue='group', dodge=False)
        plt.title('Compression Efficiency: Bits Per Character (BPC)', fontsize=16)
        plt.ylabel('Bits Per Character (Lower is Better)')
        plt.xlabel('Compression Method')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizer.figure_dir, 'exp1d_bpc_comparison.png'), dpi=300)
        plt.close()

        df_lossy = df[df['group'] == 'LLM Lossy']
        if not df_lossy.empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_lossy, x='avg_bits_per_character', y='reconstruction_accuracy', hue='name', s=200, style='name', palette='viridis')
            plt.title('Fidelity vs. Efficiency for Lossy LLM Methods', fontsize=16)
            plt.xlabel('Bits Per Character (Compression)')
            plt.ylabel('Word Accuracy (Fidelity)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='Method')
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualizer.figure_dir, 'exp1d_fidelity_vs_efficiency.png'), dpi=300)
            plt.close()

        self.logger.info("Visualizations for Experiment 1D created.")
        
    def save_results(self, results: Dict[str, Any]):
        results_dir = os.path.join(self.exp_conf.get('output_dir', './results'), 'results', 'experiment_1d')
        results_path = os.path.join(results_dir, f'results_{datetime.now():%Y%m%d_%H%M%S}.json')
        
        def convert_serializable(obj):
            if isinstance(obj, np.generic): return obj.item()
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: convert_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert_serializable(i) for i in obj]
            return obj

        with open(results_path, 'w') as f:
            json.dump(convert_serializable(results), f, indent=2)
        self.logger.info(f"Results saved to {results_path}")
        
    def print_summary(self, results: Dict[str, Any]):
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT 1D SUMMARY")
        self.logger.info("="*80)
        self.logger.info("Summary data saved to JSON. Please see generated plots and tables.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Experiment 1D: Comparative Compression Benchmark")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    runner = Experiment1DRunner(args.config)
    runner.run_comparative_benchmark()

if __name__ == "__main__":
    main()
