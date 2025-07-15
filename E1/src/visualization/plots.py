# src/visualization/plots.py
"""Visualization utilities for compression experiments."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class CompressionVisualizer:
    def __init__(
        self,
        style: str = "seaborn-v0_8-darkgrid",
        figure_dir: str = "./results/figures",
    ):
        plt.style.use(style)
        self.figure_dir = figure_dir
        import os

        os.makedirs(figure_dir, exist_ok=True)

    def plot_compression_vs_accuracy(
        self, results: Dict[str, Dict], save_name: str = "compression_vs_accuracy"
    ):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        metrics_to_plot = [
            ("compression_ratio", "Compression Ratio"),
            ("word_accuracy", "Word Accuracy"),
            ("semantic_similarity", "Semantic Similarity"),
            ("rouge1_fmeasure", "ROUGE-1 F1 Score"),
        ]
        
        # Determine shared x-axis label
        x_label = "Hyperparameter (Masking Prob / Bits / Codebook Size)"
        
        for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
            ax = axes[idx]
            for model_name, model_results in results.items():
                hyperparams = []
                metric_values = []
                for param, metrics in model_results.items():
                    if isinstance(param, (float, int)):
                        hyperparams.append(param)
                        metric_values.append(metrics.get(metric_key, 0))
                
                if hyperparams:
                    sorted_data = sorted(zip(hyperparams, metric_values))
                    hyperparams, metric_values = zip(*sorted_data)
                    ax.plot(
                        hyperparams,
                        metric_values,
                        marker="o",
                        label=model_name,
                        linewidth=2,
                    )

            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(metric_label, fontsize=12)
            ax.set_title(f"{metric_label} vs. Hyperparameter", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle("Compression Performance Metrics", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_interactive_comparison(
        self, results: Dict[str, Dict], save_name: str = "interactive_comparison"
    ):
        data_frames = []
        for model_name, model_results in results.items():
            for param, metrics in model_results.items():
                if isinstance(param, (float, int)):
                    record = {
                        "Model": model_name,
                        "Hyperparameter": param,
                        "Compression Ratio": metrics.get("compression_ratio", 0),
                        "Word Accuracy": metrics.get("word_accuracy", 0),
                        "Semantic Similarity": metrics.get(
                            "semantic_similarity", 0
                        ),
                        "ROUGE-1 F1": metrics.get("rouge1_fmeasure", 0),
                        "BERT Score F1": metrics.get("bert_score_f1", 0),
                    }
                    if 'lsq' in model_name.lower():
                        record['Parameter Type'] = 'Quantization Bits'
                    elif 'vq' in model_name.lower():
                        record['Parameter Type'] = 'Codebook Size'
                    else:
                        record['Parameter Type'] = 'Masking Probability'
                    data_frames.append(record)

        if not data_frames:
            print("No data for interactive plot.")
            return

        df = pd.DataFrame(data_frames)
        fig = px.line(df, x="Hyperparameter", y="Semantic Similarity", color="Model",
                      facet_col="Parameter Type", markers=True,
                      title="Interactive Compression Performance Comparison")
        fig.update_xaxes(matches=None) # Unlink x-axes
        fig.write_html(f"{self.figure_dir}/{save_name}.html")
        
    def plot_efficiency_vs_fidelity_tradeoff(
        self,
        results: Dict[str, Dict],
        fidelity_metric: str = "semantic_similarity",
        efficiency_metric: str = "bits_per_character",
        save_name: str = "efficiency_fidelity_tradeoff",
    ):
        plt.figure(figsize=(12, 8))
        
        for model_name, model_results in results.items():
            efficiency_values = []
            fidelity_values = []
            
            for param, metrics in model_results.items():
                if isinstance(param, (float, int)):
                    efficiency_values.append(metrics.get(efficiency_metric, 0))
                    fidelity_values.append(metrics.get(fidelity_metric, 0))

            if efficiency_values and fidelity_values:
                # Sort by efficiency for a clean line plot
                sorted_data = sorted(zip(efficiency_values, fidelity_values))
                efficiency_values, fidelity_values = zip(*sorted_data)
                
                plt.plot(
                    efficiency_values,
                    fidelity_values,
                    marker="o",
                    linestyle="--",
                    label=model_name,
                    linewidth=2,
                    markersize=8,
                    alpha=0.8
                )

        plt.xlabel(efficiency_metric.replace("_", " ").title() + " (Lower is Better)", fontsize=12)
        plt.ylabel(fidelity_metric.replace("_", " ").title() + " (Higher is Better)", fontsize=12)
        plt.title(f"{fidelity_metric.replace('_', ' ').title()} vs. {efficiency_metric.replace('_', ' ').title()}", fontsize=16)
        plt.legend(title="Model & Method")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().invert_xaxis() # Lower bits/char is better, so plot from right to left
        
        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_heatmap_comparison(
        self,
        results: Dict[str, Dict],
        metric: str = "semantic_similarity",
        save_name: str = "heatmap_comparison",
    ):
        models = list(results.keys())
        # This will need to be smarter if params are different types
        try:
            hyperparams = sorted(
                list(set(p for model in results.values() for p in model if isinstance(p, (float, int))))
            )
        except (IndexError, KeyError):
            print("Could not generate heatmap, result structure may be inconsistent.")
            return

        matrix = np.zeros((len(models), len(hyperparams)))

        for i, model in enumerate(models):
            for j, param in enumerate(hyperparams):
                matrix[i, j] = results.get(model, {}).get(param, {}).get(metric, 0)

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            matrix,
            xticklabels=[f"{p:.2f}" for p in hyperparams],
            yticklabels=models,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": metric.replace("_", " ").title()},
        )

        plt.xlabel("Hyperparameter Value", fontsize=12)
        plt.ylabel("Model & Method", fontsize=12)
        plt.title(f'{metric.replace("_", " ").title()} Heatmap', fontsize=14)

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_bits_per_token_analysis(
        self, results: Dict[str, Dict], save_name: str = "bits_per_token"
    ):
        plt.figure(figsize=(12, 8))
        for model_name, model_results in results.items():
            hyperparams = []
            bpt_values = []
            for param, metrics in model_results.items():
                if isinstance(param, (float, int)):
                    hyperparams.append(param)
                    # Approximate BPT from BPC, assuming avg 5 chars/token
                    bpt_values.append(metrics.get("bits_per_character", 0) * 5)
            
            if hyperparams:
                sorted_data = sorted(zip(hyperparams, bpt_values))
                hyperparams, bpt_values = zip(*sorted_data)

                plt.plot(
                    hyperparams, bpt_values, marker="o", label=model_name, linewidth=2
                )

        plt.xlabel("Hyperparameter Value", fontsize=12)
        plt.ylabel("Approx. Bits Per Token", fontsize=12)
        plt.title("Compression Efficiency: Bits Per Token", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(
            y=32, color="red", linestyle="--", alpha=0.5, label="Uncompressed Token (FP32 ID)"
        )
        plt.axhline(
            y=8, color="green", linestyle="--", alpha=0.5, label="Gzip-like BPT"
        )

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_model_comparison_radar(
        self,
        results: Dict[str, Dict],
        hyperparameter_val: float = 0.5,
        save_name: str = "radar_comparison",
    ):
        metrics = [
            "word_accuracy",
            "semantic_similarity",
            "rouge1_fmeasure",
            "bert_score_f1",
            "compression_ratio",
        ]
        metric_labels = [
            "Word\nAccuracy",
            "Semantic\nSimilarity",
            "ROUGE-1\nF1",
            "BERT Score\nF1",
            "Compression\nRatio",
        ]

        fig = go.Figure()

        for model_name, model_results in results.items():
            # Find the closest hyperparameter
            valid_keys = [k for k in model_results.keys() if isinstance(k, (int, float))]
            if not valid_keys: continue
            closest_param = min(valid_keys, key=lambda x:abs(x-hyperparameter_val))
            
            if closest_param in model_results:
                values = []
                for metric in metrics:
                    value = model_results[closest_param].get(metric, 0)
                    if metric == "compression_ratio":
                        # Normalize compression ratio for plotting: 0=1x, 1=10x or more
                        value = min(value / 10, 1.0)
                    values.append(value)

                fig.add_trace(
                    go.Scatterpolar(
                        r=values, theta=metric_labels, fill="toself", name=f"{model_name} (param~{closest_param:.2f})"
                    )
                )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            showlegend=True,
            title=f"Model Comparison at Hyperparameter ~{hyperparameter_val}",
        )

        fig.write_html(f"{self.figure_dir}/{save_name}.html")

    def plot_reconstruction_examples(
        self, examples: List[Dict[str, str]], save_name: str = "reconstruction_examples"
    ):
        if not examples: return
        fig, axes = plt.subplots(len(examples), 1, figsize=(14, 4 * len(examples)))
        if len(examples) == 1:
            axes = [axes]

        for idx, (ax, example) in enumerate(zip(axes, examples)):
            original = example["original"]
            reconstructed = example["reconstructed"]

            ax.text(0.02, 0.9, "Original:", transform=ax.transAxes, fontweight="bold")
            ax.text(0.02, 0.7, original, transform=ax.transAxes, wrap=True, fontsize=10)

            ax.text(
                0.02, 0.5, "Reconstructed:", transform=ax.transAxes, fontweight="bold"
            )
            ax.text(
                0.02, 0.3, reconstructed, transform=ax.transAxes, wrap=True, fontsize=10
            )

            ax.text(
                0.02,
                0.1,
                f"Model: {example['model']} | Param: {example['param']}",
                transform=ax.transAxes,
                fontsize=9,
                style="italic",
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        plt.suptitle("Reconstruction Examples", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def create_summary_report(
        self, results: Dict[str, Dict], save_name: str = "summary_report"
    ):
        if not results: return
        self.plot_compression_vs_accuracy(results, f"{save_name}_accuracy")
        self.plot_interactive_comparison(results, f"{save_name}_interactive")
        self.plot_heatmap_comparison(
            results, "semantic_similarity", f"{save_name}_heatmap_semantic"
        )
        self.plot_heatmap_comparison(
            results, "compression_ratio", f"{save_name}_heatmap_compression"
        )
        self.plot_bits_per_token_analysis(results, f"{save_name}_bits_per_token")
        self.plot_efficiency_vs_fidelity_tradeoff(results, save_name=f"{save_name}_tradeoff")
        self._create_summary_table(results, f"{save_name}_table")

    def _create_summary_table(self, results: Dict[str, Dict], save_name: str):
        summary_data = []
        for model_name, model_results in results.items():
            for param, metrics in model_results.items():
                if isinstance(param, (float, int)):
                    row = {
                        "Model & Method": model_name,
                        "Hyperparameter": f"{param:.2f}",
                        "Compression Ratio": f"{metrics.get('compression_ratio', 0):.2f}",
                        "Word Accuracy": f"{metrics.get('word_accuracy', 0):.3f}",
                        "Semantic Similarity": f"{metrics.get('semantic_similarity', 0):.3f}",
                        "ROUGE-1 F1": f"{metrics.get('rouge1_fmeasure', 0):.3f}",
                        "BERT Score F1": f"{metrics.get('bert_score_f1', 0):.3f}",
                        "Bits/Char": f"{metrics.get('bits_per_character', 0):.3f}"
                    }
                    if 'lsq' in model_name.lower():
                        row['Hyperparameter'] = str(metrics.get('quantization_bits'))
                    elif 'vq' in model_name.lower():
                        row['Hyperparameter'] = str(param)
                    summary_data.append(row)
        if not summary_data: return

        df = pd.DataFrame(summary_data)
        fig, ax = plt.subplots(figsize=(16, len(summary_data) * 0.4 + 2))
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(
            cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        plt.title("Compression Experiment Results Summary", fontsize=16, pad=20)
        plt.savefig(f"{self.figure_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

