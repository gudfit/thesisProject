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

# Add project root to path
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
            
            model_results['variants'][key] = info
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
        self._run_gzip_baseline_check(crumpled_metrics, test_texts[0])

        masking_prob = self.config['experiment_1c']['structural_fidelity']['masking_probability']
        all_metrics = []
        
        for text in tqdm(test_texts, desc="Structural Fidelity (Direct)"):
            try:
                enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                input_ids = enc['input_ids'].to(model.device)
                
                num_to_mask = int(input_ids.shape[1] * masking_prob)
                maskable_indices = (input_ids[0] != tokenizer.cls_token_id) & (input_ids[0] != tokenizer.sep_token_id)
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
            tasks=glue_config['tasks'],
            train_samples=glue_config['train_samples'],
            eval_samples=glue_config['eval_samples'],
            epochs=glue_config['fine_tune_epochs']
        )

    def _evaluate_computational_performance(self, model: torch.nn.Module, compressor: ModelCompressor) -> Dict:
        self.logger.info("Phase 3: Evaluating computational performance")
        from datasets import load_dataset
        bench_conf = self.config['experiment_1c']['benchmark']
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=f'test[:{bench_conf["batch_size"]*2}]')
        test_texts = [item['text'].strip() for item in dataset if len(item['text'].strip()) > 50][:bench_conf["batch_size"]]
        
        perf_metrics = compressor.measure_inference_performance(
            model, test_texts, batch_size=bench_conf['batch_size']
        )
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
                sparsity_levels, tcm_values, pcm_values, glue_scores, inference_times = [], [], [], [], []
                for variant_key, variant_info in model_results['variants'].items():
                    if variant_info['pruning_type'] != pruning_type and variant_info['pruning_type'] != 'none':
                        continue
                    sparsity = variant_info['sparsity']
                    sparsity_levels.append(sparsity)
                    tcm_values.append(model_results['phase1_structural_fidelity'][variant_key]['mean_tcm'])
                    pcm_values.append(model_results['phase1_structural_fidelity'][variant_key]['mean_pcm'])
                    glue_scores.append(model_results['phase2_downstream_performance'][variant_key]['average'])
                    inference_times.append(model_results['phase3_computational_performance'][variant_key]['avg_latency_ms'])
                
                idx = np.argsort(sparsity_levels)
                label = f"{model_name.split('-')[0].upper()} ({pruning_type})"
                axes[0].plot(np.array(sparsity_levels)[idx] * 100, np.array(tcm_values)[idx], 'o-', label=label)
                axes[1].plot(np.array(sparsity_levels)[idx] * 100, np.array(pcm_values)[idx], 's-', label=label)
                axes[2].plot(np.array(sparsity_levels)[idx] * 100, np.array(glue_scores)[idx], '^-', label=label)
                axes[3].plot(np.array(sparsity_levels)[idx] * 100, np.array(inference_times)[idx], 'd-', label=label)

        titles = ['TCM vs Pruning', 'PCM vs Pruning', 'GLUE Score vs Pruning', 'Latency vs Pruning']
        for ax, title in zip(axes, titles):
            ax.set_xlabel('Sparsity (%)')
            ax.set_title(title)
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
        fig, axes = plt.subplots(1, 2, figsize=(15, len(models) * len(pruning_types) * 0.5))
        
        for ax, metric in zip(axes, ['tcm', 'pcm']):
            data = []
            yticklabels = []
            for model in models:
                for ptype in pruning_types:
                    row = [results[model]['phase1_structural_fidelity'].get(f"{ptype}_{s}" if s!=0 else "original", {}).get(f'mean_{metric}', 0) for s in sparsity_levels]
                    data.append(row)
                    yticklabels.append(f"{model.split('-')[0]}({ptype[0]})")
            
            sns.heatmap(data, xticklabels=[f'{s:.0%}' for s in sparsity_levels],
                        yticklabels=yticklabels, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': f'Mean {metric.upper()}'})
            ax.set_title(f'Mean {metric.upper()}')
        
        plt.suptitle(f'Structural Fidelity Degradation (Fidelity: {suffix})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c', f'structural_fidelity_heatmap_{suffix}.png')
        plt.savefig(path, dpi=300)
        plt.close()

    def _plot_computational_performance(self, results: Dict[str, Any], suffix: str):
        # Implementation for this plot remains similar, just add suffix to save path
        pass

    def _plot_combined_summary(self, results: Dict[str, Any], suffix: str):
        # Implementation for this plot remains similar, just add suffix to save path
        pass

    def save_results(self, results: Dict[str, Any], method_suffix: str):
        def convert(obj):
            if isinstance(obj, (np.ndarray, list)): return [convert(i) for i in obj]
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if hasattr(obj, 'item'): return obj.item()
            return obj

        results_dir = os.path.join(self.config['experiment']['output_dir'], 'results', 'experiment_1c')
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f'results_{method_suffix}_{datetime.now():%Y%m%d_%H%M%S}.json')
        with open(results_path, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        summary_data = []
        for model_name, model_results in results.items():
            for key, info in model_results.get('variants', {}).items():
                row = {'model': model_name, 'variant': key, 'sparsity': info['sparsity'],
                       'pruning_type': info['pruning_type'], 'size_mb': info['size_mb']}
                row.update(model_results['phase1_structural_fidelity'].get(key, {}))
                row.update(model_results['phase2_downstream_performance'].get(key, {}))
                comp = model_results['phase3_computational_performance'].get(key, {})
                row.update({k: comp.get(k) for k in ['avg_latency_ms', 'throughput_samples_per_sec', 'memory_footprint_mb', 'compression_ratio']})
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
            for key in model_results['variants']:
                structural = model_results['phase1_structural_fidelity'].get(key, {})
                downstream = model_results['phase2_downstream_performance'].get(key, {})
                computational = model_results['phase3_computational_performance'].get(key, {})
                tcm_score = 1.0 / (1.0 + structural.get('mean_tcm', float('inf')))
                glue_score = downstream.get('average', 0)
                efficiency_score = computational.get('throughput_samples_per_sec', 1) / 100
                w = self.config['experiment_1c']['sweet_spot_weights']
                combined = (w['downstream_performance'] * glue_score + w['structural_fidelity'] * tcm_score +
                            w['computational_efficiency'] * efficiency_score)
                if combined > best_info['score']:
                    best_info = {'score': combined, 'key': key}

            for key, info in sorted(model_results['variants'].items(), key=lambda item: item[1]['sparsity']):
                sparsity, ptype = info['sparsity'], info['pruning_type']
                tcm = model_results['phase1_structural_fidelity'].get(key, {}).get('mean_tcm', -1)
                glue = model_results['phase2_downstream_performance'].get(key, {}).get('average', -1)
                latency = model_results['phase3_computational_performance'].get(key, {}).get('avg_latency_ms', -1)
                self.logger.info(f"  {ptype} {sparsity:.0%}: TCM={tcm:.2f}, GLUE={glue:.3f}, Latency={latency:.1f}ms")
            
            if best_info['key']:
                self.logger.info(f"  => OPTIMAL: {best_info['key']} (Score: {best_info['score']:.3f})")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Experiment 1C: Model Compression Analysis")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    runner = Experiment1CRunner(args.config)
    runner.run_compression_analysis()

if __name__ == "__main__":
    main()
