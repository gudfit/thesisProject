# src/analysis/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ExperimentVisualizer:

    def __init__(self, style: str = "publication"):
        self.setup_style(style)
        self.colors = self._get_color_palette()

    def setup_style(self, style: str):
        if style == "publication":
            plt.rcParams.update(
                {
                    "font.size": 12,
                    "font.family": "serif",
                    "axes.linewidth": 1.2,
                    "axes.labelsize": 14,
                    "axes.titlesize": 16,
                    "xtick.labelsize": 12,
                    "ytick.labelsize": 12,
                    "legend.fontsize": 11,
                    "figure.titlesize": 18,
                    "lines.linewidth": 2,
                    "lines.markersize": 8,
                    "grid.alpha": 0.3,
                    "axes.grid": True,
                    "savefig.dpi": 300,
                    "savefig.bbox": "tight",
                }
            )
        else:
            plt.style.use(style)

    def _get_color_palette(self) -> Dict[str, str]:
        return {
            "GPT": "#1f77b4",
            "LLaMA": "#ff7f0e",
            "Cerebras": "#2ca02c",
            "Mistral": "#d62728",
            "Qwen": "#9467bd",
            "DeepSeek": "#8c564b",
            "Phi": "#e377c2",
            "Gemma": "#7f7f7f",
            "Huffman": "#bcbd22",
            "LZW": "#17becf",
            "Other": "#17becf",
        }

    def plot_performance_comparison(
        self, summary_df: pd.DataFrame, output_path: Path, metric: str = "success_rate"
    ):
        fig, ax = plt.subplots(figsize=(12, 6))
        families = summary_df.index.map(self._extract_family)
        colors = [self.colors.get(f, self.colors["Other"]) for f in families]
        bars = ax.bar(
            range(len(summary_df)), summary_df[metric], color=colors, alpha=0.8
        )
        ax.set_xlabel("Model", fontweight="bold")
        ax.set_ylabel(metric.replace("_", " ").title(), fontweight="bold")
        ax.set_title(
            f"Model Performance Comparison: {metric}", fontweight="bold", pad=20
        )
        ax.set_xticks(range(len(summary_df)))
        ax.set_xticklabels(summary_df.index, rotation=45, ha="right")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_retrieval_degradation(self, pivot_df: pd.DataFrame, output_path: Path):
        fig, ax = plt.subplots(figsize=(12, 8))

        for model in pivot_df.index:
            family = self._extract_family(model)
            color = self.colors.get(family, self.colors["Other"])

            ax.plot(
                pivot_df.columns,
                pivot_df.loc[model],
                label=model,
                color=color,
                marker="o",
                linewidth=2,
            )

        ax.set_xlabel("Retrieval Budget θ (Prompt Length)", fontweight="bold")
        ax.set_ylabel("Semantic Success Rate", fontweight="bold")
        ax.set_title("Retrieval Performance Degradation", fontweight="bold", pad=20)
        ax.set_ylim(0, 1.05)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_efficiency_scatter(self, efficiency_df: pd.DataFrame, traditional_eirs: Dict[str, float], output_path: Path):
        fig, ax = plt.subplots(figsize=(10, 8))

        families = efficiency_df.index.map(self._extract_family)

        for family in set(families):
            mask = families == family
            family_data = efficiency_df[mask]

            ax.scatter(
                family_data["effective_param_gb"],
                family_data["eir_success"],
                label=family,
                color=self.colors.get(family, self.colors["Other"]),
                s=100,
                alpha=0.8,
                edgecolors="black",
            )

        
        for comp, eir in traditional_eirs.items():
            ax.axhline(y=eir, color=self.colors.get(comp, self.colors["Other"]), linestyle='--', label=f'{comp} (Traditional)')

        ax.set_xlabel("Storage Cost (GB)", fontweight="bold")
        ax.set_ylabel("Effective Information Ratio (EIR)", fontweight="bold")
        ax.set_title(
            "Model Efficiency: EIR vs Storage (with Traditional Baselines)", fontweight="bold", pad=20
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def _extract_family(model_name: str) -> str:
        model_lower = model_name.lower()
        families = {
            "gpt": "GPT",
            "llama": "LLaMA",
            "cerebras": "Cerebras",
            "mistral": "Mistral",
            "qwen": "Qwen",
            "deepseek": "DeepSeek",
            "phi": "Phi",
            "gemma": "Gemma",
        }

        for key, family in families.items():
            if key in model_lower:
                return family
        return "Other"



