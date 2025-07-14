import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentAnalyzer:

    def __init__(self, results_df: pd.DataFrame, config: Any):
        self.results_df = results_df
        self.config = config
        self._prepare_dataframe()

    def _prepare_dataframe(self):
        if "is_semantically_equivalent" not in self.results_df.columns:
            self.results_df["is_semantically_equivalent"] = (
                self.results_df["semantic_similarity"] >= self.config.semantic_threshold
            )

    def compute_summary_statistics(self, theta: Optional[int] = None) -> pd.DataFrame:
        if theta is None:
            theta = max(self.config.theta_budgets)

        df_filtered = self.results_df[self.results_df["prompt_len_theta"] == theta]

        summary = (
            df_filtered.groupby("model_name")
            .agg(
                {
                    "is_semantically_equivalent": "mean",
                    "semantic_similarity": ["mean", "std"],
                    "retrieval_cost_ms": ["mean", "std"],
                    "is_perfect": "mean",
                }
            )
            .round(4)
        )

        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary = summary.rename(
            columns={
                "is_semantically_equivalent_mean": "success_rate",
                "semantic_similarity_mean": "avg_similarity",
                "semantic_similarity_std": "std_similarity",
                "retrieval_cost_ms_mean": "avg_latency_ms",
                "retrieval_cost_ms_std": "std_latency_ms",
                "is_perfect_mean": "perfect_rate",
            }
        )

        return summary

    def compute_retrieval_degradation(self) -> pd.DataFrame:
        pivot = self.results_df.pivot_table(
            index="model_name",
            columns="prompt_len_theta",
            values="is_semantically_equivalent",
            aggfunc="mean",
        )
        return pivot.round(4)


class FinetuneAnalyzer(ExperimentAnalyzer):

    def compute_model_efficiency(self) -> pd.DataFrame:
        summary = self.compute_summary_statistics()

        storage_costs = self.results_df.groupby("model_name")[
            "storage_cost_lambda"
        ].first()
        summary["storage_gb"] = storage_costs / 1e9
        summary["success_per_gb"] = summary["success_rate"] / summary["storage_gb"]
        summary["similarity_per_gb"] = summary["avg_similarity"] / summary["storage_gb"]
        return summary


class PruningAnalyzer(ExperimentAnalyzer):

    def compute_pruning_impact(self) -> pd.DataFrame:
        summary = self.compute_summary_statistics()
        summary["pruning_pct"] = (
            summary.index.str.extract(r"(\d+)%").astype(float).fillna(0)
        )
        baseline_perf = summary.loc[summary["pruning_pct"] == 0, "success_rate"].iloc[0]
        summary["relative_performance"] = summary["success_rate"] / baseline_perf
        return summary.sort_values("pruning_pct")
