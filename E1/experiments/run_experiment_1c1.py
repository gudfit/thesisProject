# src/experiments/run_experiment_1c.py
"""Run Experiment 1C: Effects of Pruning and Quantization on Knowledge Compression."""

import os
import sys
import yaml
import json
import logging
import gzip
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.compression_techniques import ModelCompressor
from src.evaluation.crumpled_paper_metrics import CrumpledPaperMetrics
from src.evaluation.glue_evaluator import GLUEEvaluator
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
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
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
        
        fidelity_methods = self.config['experiment_1c']['structural_fidelity'].get('method', ['direct_mlm'])
        if isinstance(fidelity_methods, str):
            fidelity_methods = [fidelity_methods]

        for fidelity_method in fidelity_methods:
            self.logger.info(f"\n{'#'*70}")
            self.logger.info(f"# Running full analysis with Structural Fidelity Method: '{fidelity_method}'")
            self.logger.info(f"{'#'*70}")
            
            results = {}
            pruning_levels = self.config['experiment_1c']['pruning']['levels']
            pruning_types = self.config['experiment_1c']['pruning']['type']

            for model_config in self.config['models']:
                model_name = model_config['name']
                self.logger.info(f"\n{'='*60}\nAnalyzing compression for {model_name}\n{'='*60}")
                compressor = ModelCompressor(model_name, self.config['experiment']['device'])
                variants = self._create_pruning_variants(compressor, pruning_types, pruning_levels)
                model_results = self._evaluate_all_variants(variants, model_name, compressor, fidelity_method)
                results[model_name] = model_results
            
            self.create_visualizations(results, fidelity_method)
            self.save_results(results, fidelity_method)
            self.print_summary(results, fidelity_method)

        self.logger.info("\nAll configured Experiment 1C runs completed!")

    def _create_pruning_variants(self, compressor: ModelCompressor, pruning_types: List[str], pruning_levels: List[float]) -> Dict:
        variants = {}
        original = compressor.load_original_model()
        variants['original'] = {
            'model': original, 'type': 'original', 'sparsity': 0.0,
            'size_mb': compressor._calculate_model_size(original), 'pruning_type': 'none'
        }
        for ptype in pruning_types:
            for sparsity in pruning_levels:
                if sparsity == 0.0: continue
                self.logger.info(f"Creating {ptype} pruned variant with {sparsity:.0%} sparsity...")
                if ptype == "magnitude":
                    pruned = compressor.apply_magnitude_pruning(original, sparsity)
                elif ptype == "structured":
                    pruned = compressor.apply_structured_pruning(original, sparsity)
                else:
                    self.logger.error(f"Unknown pruning type: {ptype}")
                    continue
                
                variants[f"{ptype}_{sparsity}"] = {
                    'model': pruned, 'type': 'pruned', 'sparsity': sparsity,
                    'size_mb': compressor._calculate_model_size(pruned), 'pruning_type': ptype
                }
        return variants
        
    def _evaluate_all_variants(self, variants: Dict, model_name: str, compressor: ModelCompressor, fidelity_method: str) -> Dict:
        model_results = {
            'variants': {}, 'phase1_structural_fidelity': {},
            'phase2_downstream_performance': {}, 'phase3_computational_performance': {}
        }
        for key, info in variants.items():
            self.logger.info(f"\n--- Evaluating variant: {key} ---")
            
            if fidelity_method == 'predictive_masking':
                structural_results = self._evaluate_structural_fidelity_pm(info['model'], model_name)
            else: 
                structural_results = self._evaluate_structural_fidelity_direct(info['model'], compressor.tokenizer)

            downstream_results = self._evaluate_downstream_performance(info['model'], model_name)
            comp_results = self._evaluate_computational_performance(info['model'], compressor)
            
            serializable_info = {k: v for k, v in info.items() if k != 'model'}
            model_results['variants'][key] = serializable_info
            model_results['phase1_structural_fidelity'][key] = structural_results
            model_results['phase2_downstream_performance'][key] = downstream_results
            model_results['phase3_computational_performance'][key] = comp_results
        
        return model_results

    def _get_fidelity_test_texts(self) -> List[str]:
        from datasets import load_dataset
        fidelity_conf = self.config['experiment_1c']['structural_fidelity']
        num_samples = fidelity_conf.get('test_samples', 20)
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=f'test[:{num_samples*2}]')
        return [item['text'] for item in dataset if len(item['text'].strip()) > 50][:num_samples]

    def _evaluate_structural_fidelity_pm(self, model: torch.nn.Module, model_name: str) -> Dict:
        self.logger.info("Phase 1: Evaluating structural fidelity (via Predictive Masking)")
        from src.models.predictive_masking import PredictiveMaskingCompressor
        
        crumpled_metrics = CrumpledPaperMetrics()
        test_texts = self._get_fidelity_test_texts()
        if test_texts:
            self._run_gzip_baseline_check(crumpled_metrics, test_texts[0])

        temp_compressor = PredictiveMaskingCompressor(model_name)
        temp_compressor.model = model
        
        masking_prob = self.config['experiment_1c']['structural_fidelity']['masking_probability']
        all_metrics = []
        for text in tqdm(test_texts, desc="Structural Fidelity (PM)"):
            try:
                compressed = temp_compressor.compress(text, masking_probability=masking_prob)
                reconstructed = temp_compressor.decompress(compressed)
                metrics = crumpled_metrics.calculate_crease_metrics(text, reconstructed)
                all_metrics.append(metrics)
            except Exception as e:
                self.logger.warning(f"Error in structural fidelity (PM): {e}")

        return self._aggregate_fidelity_metrics(all_metrics)

    def _evaluate_structural_fidelity_direct(self, model: torch.nn.Module, tokenizer) -> Dict:
        self.logger.info("Phase 1: Evaluating structural fidelity (via Direct MLM)")
        crumpled_metrics = CrumpledPaperMetrics()
        test_texts = self._get_fidelity_test_texts()
        if test_texts:
            self._run_gzip_baseline_check(crumpled_metrics, test_texts[0])

        masking_prob = self.config['experiment_1c']['structural_fidelity']['masking_probability']
        all_metrics = []
        
        for text in tqdm(test_texts, desc="Structural Fidelity (Direct)"):
            try:
                enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                input_ids = enc['input_ids'].to(model.device)
                
                num_to_mask = int(input_ids.shape[1] * masking_prob)
                maskable_indices = (input_ids[0] != tokenizer.cls_token_id) & (input_ids[0] != tokenizer.sep_token_id)
                if maskable_indices.sum() < num_to_mask: continue
                
                mask_positions = torch.where(maskable_indices)[0][torch.randperm(maskable_indices.sum())[:num_to_mask]]
                
                masked_input = input_ids.clone()
                masked_input[0, mask_positions] = tokenizer.mask_token_id

                with torch.no_grad():
                    logits = model(masked_input).logits
                
                reconstructed_ids = masked_input.clone()
                reconstructed_ids[0, mask_positions] = logits[0, mask_positions].argmax(dim=-1)
                
                reconstructed_text = tokenizer.decode(reconstructed_ids[0], skip_special_tokens=True)
                metrics = crumpled_metrics.calculate_crease_metrics(text, reconstructed_text)
                all_metrics.append(metrics)
            except Exception as e:
                self.logger.warning(f"Error in structural fidelity (Direct): {e}")

        return self._aggregate_fidelity_metrics(all_metrics)

    def _run_gzip_baseline_check(self, crumpled_metrics: CrumpledPaperMetrics, sample_text: str):
        gzipped = gzip.compress(sample_text.encode('utf-8'))
        gunzipped = gzip.decompress(gzipped).decode('utf-8')
        gzip_metrics = crumpled_metrics.calculate_crease_metrics(sample_text, gunzipped)
        self.logger.info(f"Lossless Gzip Baseline Check: TCM={gzip_metrics['total_crease_magnitude']:.1f}, PCM={gzip_metrics['peak_crease_magnitude']:.1f} (should be 0.0)")

    def _aggregate_fidelity_metrics(self, all_metrics: List[Dict]) -> Dict:
        if not all_metrics:
            return {'mean_tcm': float('inf'), 'std_tcm': 0, 'mean_pcm': float('inf'), 'std_pcm': 0, 'mean_crease_density': 1.0}
        return {
            'mean_tcm': np.mean([m['total_crease_magnitude'] for m in all_metrics]),
            'std_tcm': np.std([m['total_crease_magnitude'] for m in all_metrics]),
            'mean_pcm': np.mean([m['peak_crease_magnitude'] for m in all_metrics]),
            'std_pcm': np.std([m['peak_crease_magnitude'] for m in all_metrics]),
            'mean_crease_density': np.mean([m['crease_density'] for m in all_metrics])
        }

    def _evaluate_downstream_performance(self, model: torch.nn.Module, model_name: str) -> Dict:
        self.logger.info("Phase 2: Evaluating downstream performance on GLUE")
        glue_config = self.config['experiment_1c']['glue']
        glue_evaluator = GLUEEvaluator(model_name, self.config['experiment']['device'])
        return glue_evaluator.evaluate_on_tasks(
            model,
            tasks=glue_config.get('tasks', []),
            train_samples=glue_config.get('train_samples', 500),
            eval_samples=glue_config.get('eval_samples', 200),
            epochs=glue_config.get('fine_tune_epochs', 5)
        )

    def _evaluate_computational_performance(self, model: torch.nn.Module, compressor: ModelCompressor) -> Dict:
        self.logger.info("Phase 3: Evaluating computational performance")
        from datasets import load_dataset
        bench_conf = self.config['experiment_1c']['benchmark']
        batch_size = bench_conf.get('batch_size', 8)
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=f'test[:{batch_size*2}]')
        test_texts = [item['text'].strip() for item in dataset if len(item['text'].strip()) > 50][:batch_size]
        
        if not test_texts:
            self.logger.warning("No suitable texts found for computational benchmark.")
            return {}

        perf_metrics = compressor.measure_inference_performance(model, test_texts, batch_size=batch_size)
        param_stats = compressor.count_nonzero_parameters(model)
        perf_metrics.update(param_stats)
        return perf_metrics
    
    def create_visualizations(self, results: Dict[str, Any], method_suffix: str):
        self.logger.info(f"Creating visualizations for '{method_suffix}'...")
        self._plot_pruning_tradeoff_curves(results, method_suffix)
        self._plot_structural_fidelity_heatmap(results, method_suffix)
        self._plot_computational_performance(results, method_suffix)
        self._plot_combined_summary(results, method_suffix)

    def _plot_pruning_tradeoff_curves(self, results: Dict[str, Any], suffix: str):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        for model_name, model_results in results.items():
            for pruning_type in self.config['experiment_1c']['pruning']['type']:
                data = {'sparsity': [], 'tcm': [], 'pcm': [], 'glue': [], 'latency': []}
                for key, info in model_results['variants'].items():
                    if info['pruning_type'] not in [pruning_type, 'none']: continue
                    data['sparsity'].append(info['sparsity'])
                    data['tcm'].append(model_results['phase1_structural_fidelity'][key]['mean_tcm'])
                    data['pcm'].append(model_results['phase1_structural_fidelity'][key]['mean_pcm'])
                    data['glue'].append(model_results['phase2_downstream_performance'][key]['average'])
                    data['latency'].append(model_results['phase3_computational_performance'][key]['avg_latency_ms'])
                
                idx = np.argsort(data['sparsity'])
                label = f"{model_name.split('-')[0].upper()} ({pruning_type})"
                axes[0].plot(np.array(data['sparsity'])[idx] * 100, np.array(data['tcm'])[idx], 'o-', label=label)
                axes[1].plot(np.array(data['sparsity'])[idx] * 100, np.array(data['pcm'])[idx], 's-', label=label)
                axes[2].plot(np.array(data['sparsity'])[idx] * 100, np.array(data['glue'])[idx], '^-', label=label)
                axes[3].plot(np.array(data['sparsity'])[idx] * 100, np.array(data['latency'])[idx], 'd-', label=label)

        titles_y = ['Mean TCM', 'Mean PCM', 'Avg GLUE Score', 'Avg Latency (ms)']
        for ax, title in zip(axes, titles_y):
            ax.set_xlabel('Sparsity (%)')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.4)
        plt.suptitle(f'Pruning Trade-off Analysis (Fidelity: {suffix})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c', f'pruning_tradeoff_curves_{suffix}.png')
        plt.savefig(path, dpi=300)
        plt.close()

    def _plot_structural_fidelity_heatmap(self, results: Dict[str, Any], suffix: str):
        models = list(results.keys())
        sparsity_levels = self.config['experiment_1c']['pruning']['levels']
        pruning_types = self.config['experiment_1c']['pruning']['type']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, len(models) * len(pruning_types) * 0.6), sharey=True)
        
        for ax, metric in zip(axes, ['tcm', 'pcm']):
            data = []
            yticklabels = []
            for model in models:
                for ptype in pruning_types:
                    row = [results[model]['phase1_structural_fidelity'].get(f"{ptype}_{s}" if s!=0 else "original", {}).get(f'mean_{metric}', 0) for s in sparsity_levels]
                    data.append(row)
                    yticklabels.append(f"{model.split('-')[0]}({ptype[:4]})")
            
            sns.heatmap(data, xticklabels=[f'{s:.0%}' for s in sparsity_levels], yticklabels=yticklabels, 
                        annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': f'Mean {metric.upper()}'})
            ax.set_title(f'Mean {metric.upper()}')
        
        plt.suptitle(f'Structural Fidelity Degradation (Method: {suffix})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c', f'structural_fidelity_heatmap_{suffix}.png')
        plt.savefig(path, dpi=300)
        plt.close()

    def _plot_computational_performance(self, results: Dict[str, Any], suffix: str):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for model_name, model_results in results.items():
            for ptype in self.config['experiment_1c']['pruning']['type']:
                data = {'sparsity': [], 'memory': [], 'throughput': [], 'ratio': []}
                for key, info in model_results['variants'].items():
                    if info['pruning_type'] not in [ptype, 'none']: continue
                    data['sparsity'].append(info['sparsity'])
                    comp = model_results['phase3_computational_performance'][key]
                    data['memory'].append(comp.get('memory_footprint_mb', 0))
                    data['throughput'].append(comp.get('throughput_samples_per_sec', 0))
                    data['ratio'].append(comp.get('compression_ratio', 1.0))
                
                idx = np.argsort(data['sparsity'])
                label = f"{model_name.split('-')[0].upper()} ({ptype})"
                axes[0].plot(np.array(data['sparsity'])[idx] * 100, np.array(data['memory'])[idx], 'o-', label=label)
                axes[1].plot(np.array(data['sparsity'])[idx] * 100, np.array(data['throughput'])[idx], 's-', label=label)
                axes[2].plot(np.array(data['sparsity'])[idx] * 100, np.array(data['ratio'])[idx], '^-', label=label)

        titles = ['Memory Usage (MB)', 'Inference Throughput (samples/sec)', 'Actual Compression Ratio']
        for ax, title in zip(axes, titles):
            ax.set_xlabel('Sparsity (%)')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.4)
        plt.suptitle(f'Computational Performance vs Pruning (Method: {suffix})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c', f'computational_performance_{suffix}.png')
        plt.savefig(path, dpi=300)
        plt.close()

    def _plot_combined_summary(self, results: Dict[str, Any], suffix: str):
        fig, ax = plt.subplots(figsize=(12, 8))
        weights = self.config['experiment_1c']['sweet_spot_weights']

        for model_name, model_results in results.items():
            for ptype in self.config['experiment_1c']['pruning']['type']:
                data = {'sparsity': [], 'score': []}
                for key, info in model_results['variants'].items():
                    if info['pruning_type'] not in [ptype, 'none']: continue
                    data['sparsity'].append(info['sparsity'])
                    
                    structural = model_results['phase1_structural_fidelity'][key]
                    downstream = model_results['phase2_downstream_performance'][key]
                    computational = model_results['phase3_computational_performance'][key]
                    
                    tcm = structural.get('mean_tcm', float('inf'))
                    tcm_score = 1.0 / (1.0 + tcm) if np.isfinite(tcm) else 0
                    glue_score = downstream.get('average', 0)
                    throughput = computational.get('throughput_samples_per_sec', 1)
                    efficiency_score = np.clip(throughput / 200, 0, 1) 
                    
                    combined = (weights['downstream_performance'] * glue_score +
                                weights['structural_fidelity'] * tcm_score +
                                weights['computational_efficiency'] * efficiency_score)
                    data['score'].append(combined)

                idx = np.argsort(data['sparsity'])
                sparsity_levels = np.array(data['sparsity'])[idx]
                scores = np.array(data['score'])[idx]
                label = f"{model_name.split('-')[0].upper()} ({ptype})"
                ax.plot(sparsity_levels * 100, scores, 'o-', label=label, linewidth=3)
                if len(scores) > 0:
                    best_idx = np.argmax(scores)
                    ax.scatter(sparsity_levels[best_idx] * 100, scores[best_idx],
                               s=250, marker='*', edgecolors='black', linewidth=1.5, zorder=5)

        ax.set_xlabel('Sparsity (%)', fontsize=14)
        ax.set_ylabel('Combined Performance Score', fontsize=14)
        ax.set_title(f'Sweet Spot Analysis (Fidelity: {suffix})', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c', f'sweet_spot_analysis_{suffix}.png')
        plt.savefig(path, dpi=300)
        plt.close()

    def save_results(self, results: Dict[str, Any], method_suffix: str):
        def convert(obj):
            if isinstance(obj, np.generic): return obj.item()
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            return obj

        results_dir = os.path.join(self.config['experiment']['output_dir'], 'results', 'experiment_1c')
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f'results_{method_suffix}_{datetime.now():%Y%m%d_%H%M%S}.json')
        serializable_results = convert(results)
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        summary_data = []
        for model_name, model_results in results.items():
            for key, info in model_results.get('variants', {}).items():
                row = info.copy() 
                row['model_name'] = model_name
                row['variant_key'] = key
                
                row.update(model_results['phase1_structural_fidelity'].get(key, {}))
                row.update(model_results['phase2_downstream_performance'].get(key, {}))
                comp = model_results['phase3_computational_performance'].get(key, {})
                row.update({k: comp.get(k) for k in ['avg_latency_ms', 'throughput_samples_per_sec', 'memory_footprint_mb', 'compression_ratio', 'total_parameters', 'nonzero_parameters']})
                summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = results_path.replace('.json', '_summary.csv')
            df.to_csv(csv_path, index=False)
        self.logger.info(f"Results for '{method_suffix}' saved.")

    def print_summary(self, results: Dict[str, Any], method_suffix: str):
        self.logger.info("\n" + "="*80 + f"\nEXPERIMENT 1C SUMMARY (Fidelity: {method_suffix})\n" + "="*80)
        for model_name, model_results in results.items():
            self.logger.info(f"\n{model_name.upper()}:\n" + "-" * 40)
            best_info = {'score': -1, 'key': ''}
            for key in model_results.get('variants', {}):
                structural = model_results['phase1_structural_fidelity'].get(key, {})
                downstream = model_results['phase2_downstream_performance'].get(key, {})
                computational = model_results['phase3_computational_performance'].get(key, {})
                
                tcm = structural.get('mean_tcm', float('inf'))
                tcm_score = 1.0 / (1.0 + tcm) if np.isfinite(tcm) else 0
                glue_score = downstream.get('average', 0)
                throughput = computational.get('throughput_samples_per_sec', 1)
                efficiency_score = np.clip(throughput / 200, 0, 1)
                
                w = self.config['experiment_1c']['sweet_spot_weights']
                combined = (w['downstream_performance'] * glue_score + w['structural_fidelity'] * tcm_score +
                            w['computational_efficiency'] * efficiency_score)
                if combined > best_info['score']:
                    best_info = {'score': combined, 'key': key}

            for key, info in sorted(model_results['variants'].items(), key=lambda item: (item[1]['pruning_type'], item[1]['sparsity'])):
                sparsity, ptype = info['sparsity'], info['pruning_type']
                tcm = model_results['phase1_structural_fidelity'].get(key, {}).get('mean_tcm', -1)
                glue = model_results['phase2_downstream_performance'].get(key, {}).get('average', -1)
                latency = model_results['phase3_computational_performance'].get(key, {}).get('avg_latency_ms', -1)
                self.logger.info(f"  {ptype:<12} {sparsity:4.0%}: TCM={tcm:6.2f}, GLUE={glue:.3f}, Latency={latency:5.1f}ms")
            
            if best_info['key']:
                self.logger.info(f"  => OPTIMAL: {best_info['key']} (Combined Score: {best_info['score']:.3f})")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Experiment 1C: Model Compression Analysis")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    runner = Experiment1CRunner(args.config)
    runner.run_compression_analysis()

if __name__ == "__main__":
    main()
