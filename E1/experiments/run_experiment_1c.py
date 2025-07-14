# src/experiments/run_experiment_1c.py
"""Run Experiment 1C: Effects of Pruning and Quantization on Knowledge Compression."""

import os
import sys
import yaml
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.compression_techniques import ModelCompressor
from src.models.knowledge_compressor import KnowledgeCompressor
from src.evaluation.crumpled_paper_metrics import CrumpledPaperMetrics
from src.evaluation.glue_evaluator import GLUEEvaluator
from src.data.lama_probe_loader import LAMAProbeLoader
from src.visualization.plots import CompressionVisualizer


class Experiment1CRunner:
    """Runner for Experiment 1C: Model Compression Analysis."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self._setup_logging()
        self._create_output_dirs()
        self.visualizer = CompressionVisualizer(
            figure_dir=os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c')
        )

    def _setup_logging(self):
        log_dir = os.path.join(self.config['experiment']['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f'experiment_1c_{datetime.now():%Y%m%d_%H%M%S}.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _create_output_dirs(self):
        dirs = [
            os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c'),
            os.path.join(self.config['experiment']['output_dir'], 'results', 'experiment_1c')
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def run_compression_analysis(self):
        self.logger.info("Starting Experiment 1C: Model Compression Analysis")
        results = {}

        pruning_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        pruning_types = self.config['experiment_1c']['pruning']['type']

        for model_config in self.config['models']:
            model_name = model_config['name']
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Analyzing compression for {model_name}")
            self.logger.info(f"{'='*60}")

            compressor = ModelCompressor(model_name, self.config['experiment']['device'])
            variants = {}

            for pruning_type in pruning_types:
                for sparsity in pruning_levels:
                    variant_key = f"{pruning_type}_{sparsity}"
                    if sparsity == 0.0:
                        original = compressor.load_original_model()
                        variants['original'] = {
                            'model': original,
                            'type': 'original',
                            'sparsity': 0.0,
                            'compression_ratio': 1.0,
                            'size_mb': compressor._calculate_model_size(original),
                            'pruning_type': 'none'
                        }
                    else:
                        self.logger.info(f"Creating {pruning_type} pruned variant with {sparsity:.0%} sparsity...")
                        if pruning_type == "magnitude":
                            pruned = compressor.apply_magnitude_pruning(compressor.load_original_model(), sparsity)
                        elif pruning_type == "structured":
                            pruned = compressor.apply_structured_pruning(compressor.load_original_model(), sparsity)
                        else:
                            raise ValueError(f"Unknown pruning type: {pruning_type}")

                        param_stats = compressor.count_nonzero_parameters(pruned)
                        variants[variant_key] = {
                            'model': pruned,
                            'type': 'pruned',
                            'sparsity': sparsity,
                            'actual_sparsity': param_stats['sparsity'],
                            'compression_ratio': param_stats['compression_ratio'],
                            'size_mb': compressor._calculate_model_size(pruned),
                            'pruning_type': pruning_type
                        }

            # Evaluate each variant
            model_results = {
                'variants': {},
                'phase1_structural_fidelity': {},
                'phase2_downstream_performance': {},
                'phase3_computational_performance': {}
            }

            for variant_key, variant_info in variants.items():
                self.logger.info(f"\nEvaluating variant: {variant_key}")

                structural_results = self.evaluate_structural_fidelity(variant_info['model'], model_name)
                model_results['phase1_structural_fidelity'][variant_key] = structural_results

                downstream_results = self.evaluate_downstream_performance(variant_info['model'], model_name)
                model_results['phase2_downstream_performance'][variant_key] = downstream_results

                computational_results = self.evaluate_computational_performance(variant_info['model'], compressor)
                model_results['phase3_computational_performance'][variant_key] = computational_results

                model_results['variants'][variant_key] = variant_info

            results[model_name] = model_results

        self.create_visualizations(results)
        self.save_results(results)
        self.print_summary(results)
        self.logger.info("\nExperiment 1C completed!")
        return results

    def evaluate_structural_fidelity(self, model, model_name: str) -> Dict[str, float]:
        crumpled_metrics = CrumpledPaperMetrics()
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test[:100]')
        test_texts = [item['text'] for item in dataset if len(item['text'].strip()) > 50][:20]
        from src.models.predictive_masking import PredictiveMaskingCompressor
        temp_compressor = PredictiveMaskingCompressor(model_name)
        temp_compressor.model = model
        all_metrics = []
        for text in test_texts:
            try:
                compressed = temp_compressor.compress(text, masking_probability=0.3)
                reconstructed = temp_compressor.decompress(compressed)
                metrics = crumpled_metrics.calculate_crease_metrics(text, reconstructed)
                all_metrics.append(metrics)
            except Exception as e:
                self.logger.warning(f"Error in structural fidelity evaluation: {e}")
        if all_metrics:
            aggregated = {
                'mean_tcm': np.mean([m['total_crease_magnitude'] for m in all_metrics]),
                'std_tcm': np.std([m['total_crease_magnitude'] for m in all_metrics]),
                'mean_pcm': np.mean([m['peak_crease_magnitude'] for m in all_metrics]),
                'std_pcm': np.std([m['peak_crease_magnitude'] for m in all_metrics]),
                'mean_crease_density': np.mean([m['crease_density'] for m in all_metrics])
            }
        else:
            aggregated = {
                'mean_tcm': float('inf'), 'std_tcm': 0.0, 'mean_pcm': float('inf'),
                'std_pcm': 0.0, 'mean_crease_density': 1.0
            }
        return aggregated

    def evaluate_downstream_performance(self, model, model_name: str) -> Dict[str, float]:
        glue_evaluator = GLUEEvaluator(model_name, self.config['experiment']['device'])
        tasks = ['sst2', 'mrpc', 'rte']
        train_samples = 1000
        eval_samples = 200
        epochs = 5
        results = glue_evaluator.evaluate_on_tasks(
            model, tasks=tasks, train_samples=train_samples, eval_samples=eval_samples, epochs=epochs
        )
        return results

    def evaluate_computational_performance(self, model, compressor: ModelCompressor) -> Dict[str, float]:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test[:500]')
        test_texts = [item['text'].strip() for item in dataset if 50 < len(item['text'].strip()) < 500][:16]
        perf_metrics = compressor.measure_inference_performance(model, test_texts, batch_size=8)
        param_stats = compressor.count_nonzero_parameters(model)
        perf_metrics.update(param_stats)
        return perf_metrics

    def create_visualizations(self, results: Dict[str, Any]):
        self._plot_pruning_tradeoff_curves(results)
        self._plot_structural_fidelity_heatmap(results)
        self._plot_computational_performance(results)
        self._plot_combined_summary(results)

    def _plot_pruning_tradeoff_curves(self, results: Dict[str, Any]):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        for model_name, model_results in results.items():
            for pruning_type in self.config['experiment_1c']['pruning']['type']:
                sparsity_levels, tcm_values, pcm_values, glue_scores, inference_times = [], [], [], [], []
                for variant_key, variant_info in model_results['variants'].items():
                    if variant_info['pruning_type'] != pruning_type and variant_info['pruning_type'] != 'none':
                        continue
                    sparsity = variant_info['sparsity']
                    sparsity_levels.append(sparsity)
                    structural = model_results['phase1_structural_fidelity'][variant_key]
                    tcm_values.append(structural['mean_tcm'])
                    pcm_values.append(structural['mean_pcm'])
                    downstream = model_results['phase2_downstream_performance'][variant_key]
                    glue_scores.append(downstream['average'])
                    computational = model_results['phase3_computational_performance'][variant_key]
                    inference_times.append(computational['avg_latency_ms'])
                idx = np.argsort(sparsity_levels)
                sparsity_levels = np.array(sparsity_levels)[idx]
                tcm_values = np.array(tcm_values)[idx]
                pcm_values = np.array(pcm_values)[idx]
                glue_scores = np.array(glue_scores)[idx]
                inference_times = np.array(inference_times)[idx]
                label = f"{model_name.split('-')[0].upper()} ({pruning_type})"
                axes[0].plot(sparsity_levels * 100, tcm_values, 'o-', label=label, linewidth=2)
                axes[1].plot(sparsity_levels * 100, pcm_values, 's-', label=label, linewidth=2)
                axes[2].plot(sparsity_levels * 100, glue_scores, '^-', label=label, linewidth=2)
                axes[3].plot(sparsity_levels * 100, inference_times, 'd-', label=label, linewidth=2)
        titles = ['Structural Degradation vs Pruning', 'Worst-Case Degradation vs Pruning',
                  'Downstream Performance vs Pruning', 'Inference Latency vs Pruning']
        for ax, title in zip(axes, titles):
            ax.set_xlabel('Sparsity (%)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle('Pruning Trade-off Analysis', fontsize=16)
        plt.tight_layout()
        path = os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c', 'pruning_tradeoff_curves.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_structural_fidelity_heatmap(self, results: Dict[str, Any]):
        models = list(results.keys())
        sparsity_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        pruning_types = self.config['experiment_1c']['pruning']['type']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for ax, metric in [(ax1, 'tcm'), (ax2, 'pcm')]:
            data = []
            for model in models:
                for ptype in pruning_types:
                    row = []
                    for sparsity in sparsity_levels:
                        key = f"{ptype}_{sparsity}" if sparsity != 0 else 'original'
                        val = results[model]['phase1_structural_fidelity'].get(key, {})
                        row.append(val.get(f'mean_{metric}', 0))
                    data.append(row)
            sns.heatmap(data, xticklabels=[f'{s:.0%}' for s in sparsity_levels],
                        yticklabels=[f"{m.split('-')[0]} ({p})" for m in models for p in pruning_types],
                        annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': metric.upper()})
            ax.set_xlabel('Sparsity Level')
            ax.set_title(f'{metric.upper()} Heatmap')
        plt.suptitle('Structural Fidelity Degradation', fontsize=14)
        plt.tight_layout()
        path = os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c', 'structural_fidelity_heatmap.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_computational_performance(self, results: Dict[str, Any]):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        pruning_types = self.config['experiment_1c']['pruning']['type']
        for model_name, model_results in results.items():
            for ptype in pruning_types:
                sparsity_levels, memory, throughput, comp_ratio = [], [], [], []
                for variant_key, variant_info in model_results['variants'].items():
                    if variant_info['pruning_type'] != ptype and variant_info['pruning_type'] != 'none':
                        continue
                    sparsity = variant_info['sparsity']
                    sparsity_levels.append(sparsity)
                    comp = model_results['phase3_computational_performance'][variant_key]
                    memory.append(comp['memory_footprint_mb'])
                    throughput.append(comp['throughput_samples_per_sec'])
                    comp_ratio.append(comp.get('compression_ratio', 1.0))
                idx = np.argsort(sparsity_levels)
                sparsity_levels = np.array(sparsity_levels)[idx]
                memory = np.array(memory)[idx]
                throughput = np.array(throughput)[idx]
                comp_ratio = np.array(comp_ratio)[idx]
                label = f"{model_name.split('-')[0].upper()} ({ptype})"
                axes[0].plot(sparsity_levels * 100, memory, 'o-', label=label, linewidth=2)
                axes[1].plot(sparsity_levels * 100, throughput, 's-', label=label, linewidth=2)
                axes[2].plot(sparsity_levels * 100, comp_ratio, '^-', label=label, linewidth=2)
        titles = ['Memory Usage vs Pruning', 'Inference Throughput vs Pruning', 'Actual Compression vs Pruning']
        for ax, title in zip(axes, titles):
            ax.set_xlabel('Sparsity (%)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle('Computational Performance Analysis', fontsize=16)
        plt.tight_layout()
        path = os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c', 'computational_performance.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_combined_summary(self, results: Dict[str, Any]):
        fig, ax = plt.subplots(figsize=(12, 8))
        pruning_types = self.config['experiment_1c']['pruning']['type']
        for model_name, model_results in results.items():
            for ptype in pruning_types:
                sparsity_levels, combined_scores = [], []
                for variant_key, variant_info in model_results['variants'].items():
                    if variant_info['pruning_type'] != ptype and variant_info['pruning_type'] != 'none':
                        continue
                    sparsity = variant_info['sparsity']
                    sparsity_levels.append(sparsity)
                    structural = model_results['phase1_structural_fidelity'][variant_key]
                    downstream = model_results['phase2_downstream_performance'][variant_key]
                    computational = model_results['phase3_computational_performance'][variant_key]
                    tcm_score = 1.0 / (1.0 + structural['mean_tcm'])
                    glue_score = downstream['average']
                    efficiency_score = computational['throughput_samples_per_sec'] / 100
                    combined = 0.4 * glue_score + 0.3 * tcm_score + 0.3 * efficiency_score
                    combined_scores.append(combined)
                idx = np.argsort(sparsity_levels)
                sparsity_levels = np.array(sparsity_levels)[idx]
                combined_scores = np.array(combined_scores)[idx]
                label = f"{model_name.split('-')[0].upper()} ({ptype})"
                ax.plot(sparsity_levels * 100, combined_scores, 'o-', label=label, linewidth=3)
                if len(sparsity_levels) > 0:
                    best_idx = np.argmax(combined_scores)
                    ax.scatter(sparsity_levels[best_idx] * 100, combined_scores[best_idx],
                               s=200, marker='*', edgecolors='black', linewidth=2)
        ax.set_xlabel('Sparsity (%)', fontsize=14)
        ax.set_ylabel('Combined Performance Score', fontsize=14)
        ax.set_title('Sweet Spot Analysis: Optimal Pruning Level', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.02, 'Combined Score = 0.4×GLUE + 0.3×(1/TCM) + 0.3×Efficiency',
                transform=ax.transAxes, fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        path = os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c', 'sweet_spot_analysis.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, results: Dict[str, Any]):
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        os.makedirs(os.path.join(self.config['experiment']['output_dir'], 'results', 'experiment_1c'), exist_ok=True)
        results_path = os.path.join(self.config['experiment']['output_dir'], 'results', 'experiment_1c',
                                    f'results_{datetime.now():%Y%m%d_%H%M%S}.json')
        with open(results_path, 'w') as f:
            json.dump(convert(results), f, indent=2)
        summary_data = []
        for model_name, model_results in results.items():
            for key, info in model_results['variants'].items():
                row = {'model': model_name, 'variant': key, 'sparsity': info['sparsity'],
                       'pruning_type': info['pruning_type'], 'size_mb': info['size_mb'],
                       'compression_ratio': info['compression_ratio']}
                row.update(model_results['phase1_structural_fidelity'][key])
                row.update(model_results['phase2_downstream_performance'][key])
                comp = model_results['phase3_computational_performance'][key]
                row.update({k: comp[k] for k in ['avg_latency_ms', 'throughput_samples_per_sec', 'memory_footprint_mb']})
                summary_data.append(row)
        df = pd.DataFrame(summary_data)
        csv_path = results_path.replace('.json', '_summary.csv')
        df.to_csv(csv_path, index=False)

    def print_summary(self, results: Dict[str, Any]):
        print("\n" + "="*80)
        print("EXPERIMENT 1C SUMMARY: PRUNING EFFECTS ON KNOWLEDGE COMPRESSION")
        print("="*80)
        for model_name, model_results in results.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 40)
            best, best_score = None, -float('inf')
            pruning_types = self.config['experiment_1c']['pruning']['type']
            for ptype in pruning_types:
                for sparsity in [0.0, 0.3, 0.5, 0.7]:
                    key = 'original' if sparsity == 0 else f"{ptype}_{sparsity}"
                    if key not in model_results['variants']: continue
                    info = model_results['variants'][key]
                    structural = model_results['phase1_structural_fidelity'][key]
                    downstream = model_results['phase2_downstream_performance'][key]
                    computational = model_results['phase3_computational_performance'][key]
                    tcm_score = 1.0 / (1.0 + structural['mean_tcm'])
                    combined = 0.4 * downstream['average'] + 0.3 * tcm_score + 0.3 * (computational['throughput_samples_per_sec'] / 100)
                    if combined > best_score:
                        best_score, best = combined, (ptype, sparsity)
                    if sparsity in [0.0, 0.3, 0.5, 0.7]:
                        print(f"  {ptype} {sparsity:.0%}: TCM={structural['mean_tcm']:.2f}, GLUE={downstream['average']:.3f}, Latency={computational['avg_latency_ms']:.1f} ms, Ratio={info['compression_ratio']:.2f}x")
            if best:
                print(f"\n  OPTIMAL ({best[0]}, {best[1]:.0%}): Combined Score={best_score:.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Experiment 1C: Model Compression Analysis")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    runner = Experiment1CRunner(args.config)
    runner.run_compression_analysis()


if __name__ == "__main__":
    main()
