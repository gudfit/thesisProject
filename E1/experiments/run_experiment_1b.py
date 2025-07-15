# experiments/run_experiment_1b.py
"""Run Experiment 1B: LLM as Static Knowledge Compressor vs Traditional Algorithms."""

import random
import argparse
import os
import sys
import yaml
import json
import time
import logging
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.knowledge_compressor import KnowledgeCompressor
from src.visualization.plots import CompressionVisualizer
from src.data.lama_probe_loader import LAMAProbeLoader
from src.data.factual_probe_generator import FactualProbeGenerator 

class Experiment1BRunner:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self._create_output_dirs()
        self._setup_logging()

        self.visualizer = CompressionVisualizer(
            figure_dir=os.path.join(
                self.config.get('experiment', {}).get('output_dir', './results'), "figures", "experiment_1b"
            )
        )

    def _setup_logging(self):
        log_dir = os.path.join(self.config.get('experiment', {}).get('output_dir', './results'), "logs")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f"experiment_1b_{datetime.now():%Y%m%d_%H%M%S}.log")),
                logging.StreamHandler(),
            ],
        )
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        self.logger = logging.getLogger(__name__)

    def _create_output_dirs(self):
        output_dir = self.config.get('experiment', {}).get('output_dir', './results')
        dirs = [
            os.path.join(output_dir, "figures", "experiment_1b"),
            os.path.join(output_dir, "results", "experiment_1b"),
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def prepare_factual_probes(self, num_probes: int, probe_source: str) -> List[Dict[str, str]]:
        self.logger.info(f"Generating {num_probes} factual probes from {probe_source} source...")

        if probe_source == "wikitext":
            probe_generator = FactualProbeGenerator()
            factual_probes = probe_generator.extract_from_wikitext(num_probes=num_probes)
        elif probe_source == "lama":
            lama_loader = LAMAProbeLoader()
            factual_probes = lama_loader.create_mixed_probe_set(num_probes)
        else: 
            probe_generator = FactualProbeGenerator()
            lama_loader = LAMAProbeLoader()
            wikitext_probes = probe_generator.extract_from_wikitext(num_probes=num_probes // 2)
            lama_probes = lama_loader.create_mixed_probe_set(num_probes - len(wikitext_probes))
            factual_probes = wikitext_probes + lama_probes
            random.shuffle(factual_probes)

        self.logger.info(f"Generated {len(factual_probes)} factual probes.")
        return factual_probes[:num_probes]

    def run_knowledge_compression_analysis(self, probe_source: str):
        self.logger.info("Starting Experiment 1B: Knowledge Compression Analysis")
        results = {}

        exp_1b_conf = self.config.get("experiment_1b", {})

        self.logger.info("Loading corpus data...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{exp_1b_conf.get('corpus_size', 5000)}]")
        corpus_texts = [item["text"] for item in dataset if len(item["text"].strip()) > 100]

        test_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"test[:{exp_1b_conf.get('test_size', 500)}]")
        test_texts = [item["text"] for item in test_dataset if len(item["text"].strip()) > 100]

        factual_probes = self.prepare_factual_probes(
            exp_1b_conf.get("factual_probes", 1000), probe_source
        )

        for model_config in self.config.get("models", []):
            model_name = model_config["name"]
            self.logger.info(f"\nAnalyzing {model_name} as a static knowledge compressor...")
            compressor = KnowledgeCompressor(model_name, self.config.get('experiment', {}).get('device', 'cuda'))

            compressor.fine_tune_on_corpus(corpus_texts, epochs=exp_1b_conf.get('fine_tune_epochs', 1))
            model_results = {"llm": {}, "gzip": {}, "comparison": {}}
            self.logger.info("Phase 1: Analyzing sizes and functional utility...")
            model_results["llm"]["size"] = compressor.calculate_model_size()
            model_results["gzip"]["size"] = compressor.compress_corpus_gzip(corpus_texts)
            model_results["llm"]["factual_recall"] = compressor.measure_factual_recall(factual_probes)
            model_results["llm"]["reasoning_capability"] = compressor.measure_reasoning_capability()
            model_results["llm"]["perplexity"] = compressor.measure_perplexity(test_texts)

            self.logger.info("Phase 2: Measuring computational costs...")
            model_results["llm"]["inference_cost"] = compressor.measure_inference_cost(test_texts)

            self.logger.info("Phase 3: Measuring data efficiency...")
            sst_dataset = load_dataset("glue", "sst2", split="train")
            fine_tune_data = [{"text": ex['sentence'], "label": ex['label']} for ex in sst_dataset]
            model_results["llm"]["data_efficiency"] = compressor.measure_data_efficiency(
                fine_tune_data, target_performance=exp_1b_conf.get('target_performance', 0.85)
            )

            llm_size = model_results["llm"]["size"]["disk_size_mb"]
            gzip_size = model_results["gzip"]["size"]["compressed_size_mb"]
            model_results["comparison"]["size_ratio"] = llm_size / gzip_size if gzip_size > 0 else float('inf')

            results[model_name] = model_results
            self._log_model_results(model_name, model_results)

        self.save_results(results)
        self.create_visualizations(results)
        self.logger.info("Experiment 1B completed!")
        return results

    def _log_model_results(self, model_name: str, results: Dict[str, any]):
        self.logger.info(f"\n--- Results for {model_name} ---")
        self.logger.info(f"  Model Size: {results.get('llm', {}).get('size', {}).get('disk_size_mb', 0):.2f} MB")
        self.logger.info(f"  Gzip Size: {results.get('gzip', {}).get('size', {}).get('compressed_size_mb', 0):.2f} MB")
        self.logger.info(f"  Size Ratio (LLM size / Gzip size): {results.get('comparison', {}).get('size_ratio', 0):.2f}x")
        self.logger.info(f"  Factual Recall: {results.get('llm', {}).get('factual_recall', 0):.3f}")
        self.logger.info(f"  Reasoning Capability (NLI Acc): {results.get('llm', {}).get('reasoning_capability', 0):.3f}")
        self.logger.info(f"  Perplexity on unseen text: {results.get('llm', {}).get('perplexity', 0):.2f}")
        self.logger.info(f"  Inference Time (per sample): {results.get('llm', {}).get('inference_cost', {}).get('avg_inference_time_ms', 0):.2f} ms")

    def save_results(self, results: Dict[str, any]):
        results_dir = os.path.join(self.config.get('experiment', {}).get('output_dir', './results'), "results", "experiment_1b")
        results_path = os.path.join(results_dir, f"results_{datetime.now():%Y%m%d_%H%M%S}.json")

        def convert_to_serializable(obj):
            if isinstance(obj, (np.ndarray, list)):
                return [convert_to_serializable(item) for item in obj]
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            if isinstance(obj, np.generic):
                return obj.item()
            return obj

        with open(results_path, "w") as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        self.logger.info(f"Results saved to {results_path}")

    def create_visualizations(self, results: Dict[str, any]):
        fig_dir = os.path.join(self.config.get('experiment', {}).get('output_dir', './results'), "figures", "experiment_1b")

        models, llm_sizes, gzip_sizes = [], [], []
        for model_name, res in results.items():
            models.append(model_name.split("-")[0].upper())
            llm_sizes.append(res.get('llm', {}).get('size', {}).get('disk_size_mb', 0))
            gzip_sizes.append(res.get('gzip', {}).get('size', {}).get('compressed_size_mb', 0))

        x = np.arange(len(models))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, llm_sizes, width, label="LLM Size (MB)", color="steelblue")
        bars2 = ax.bar(x + width / 2, gzip_sizes, width, label="Gzip Size (MB)", color="coral")
        ax.set_title("Storage Cost: LLM Parameters vs. Gzip Compression")
        ax.set_ylabel("Size (MB)")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "size_comparison.png"), dpi=300)
        plt.close()

        self.logger.info("Visualizations created!")


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1B: Knowledge Compression Analysis")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--probe-source", type=str, default="mixed", choices=["wikitext", "lama", "mixed"], help="Source for factual probes")
    args = parser.parse_args()
    runner = Experiment1BRunner(args.config)
    runner.run_knowledge_compression_analysis(probe_source=args.probe_source)

if __name__ == "__main__":
    main()
