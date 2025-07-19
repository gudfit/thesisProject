# src/analysis/analyzers.py

import pandas as pd
import numpy as np
from typing import Any, Optional
import logging
from scipy.optimize import curve_fit

from ..core.data_handler import DataHandler
from ..core.traditional_compressors import huffman_compress, lzw_compress

logger = logging.getLogger(__name__)

def capacity_scaling_law(N, Q_max, a, gamma):
    return Q_max * (1 - a * np.power(N, -gamma))

class ExperimentAnalyzer:
    def __init__(self, results_df: pd.DataFrame, config: Any):
        self.results_df = results_df
        self.config = config
        self._prepare_dataframe()

    def _prepare_dataframe(self):
        df = self.results_df
        if "is_semantically_equivalent" not in df.columns and "semantic_similarity" in df.columns:
            df["is_semantically_equivalent"] = df["semantic_similarity"] >= self.config.semantic_threshold
        if "sparsity" not in df.columns and "total_params" in df.columns and "nonzero_params" in df.columns:
            df["sparsity"] = np.where(df["total_params"] > 0, 1.0 - df["nonzero_params"] / df["total_params"], 0.0)
        if "success_composite" not in df.columns:
            if all(c in df.columns for c in ["semantic_similarity", "factual_recall", "lexical_recall"]):
                df["success_composite"] = 0.6 * df["semantic_similarity"] + 0.3 * df["factual_recall"] + 0.1 * df["lexical_recall"]
            else:
                df["success_composite"] = df.get("semantic_similarity", pd.Series(np.nan, index=df.index))
        self.results_df = df

    def compute_summary_statistics(self, theta: Optional[int] = None, domain: Optional[str] = None) -> pd.DataFrame:
        if theta is None:
            theta = max(self.config.theta_budgets)
        df = self.results_df
        df = df[df["prompt_len_theta"] == theta]
        if domain is not None and "eval_domain" in df.columns:
            df = df[df["eval_domain"] == domain]
        summary = (
            df.groupby("model_name")
            .agg(
                {
                    "is_semantically_equivalent": "mean",
                    "semantic_similarity": ["mean","std"],
                    "success_composite": ["mean","std"],
                    "retrieval_cost_ms": ["mean","std"],
                    "is_perfect": "mean",
                    "total_params": "first",
                    "nonzero_params": "first",
                    "effective_param_bytes": "first",
                    "sparsity": "first",
                    "storage_cost_bytes": "first" if "storage_cost_bytes" in df.columns else "max",
                }
            )
            .round(4)
        )
        summary.columns = ["_".join([c for c in col if c]).strip("_") for col in summary.columns.values]
        m = {
            "is_semantically_equivalent_mean":"success_rate",
            "semantic_similarity_mean":"avg_similarity",
            "semantic_similarity_std":"std_similarity",
            "success_composite_mean":"avg_success_composite",
            "success_composite_std":"std_success_composite",
            "retrieval_cost_ms_mean":"avg_latency_ms",
            "retrieval_cost_ms_std":"std_latency_ms",
            "is_perfect_mean":"perfect_rate",
            "total_params_first":"total_params",
            "nonzero_params_first":"nonzero_params",
            "effective_param_bytes_first":"effective_param_bytes",
            "sparsity_first":"sparsity",
            "storage_cost_bytes_first":"storage_cost_bytes",
        }
        summary = summary.rename(columns={k:v for k,v in m.items() if k in summary.columns})
        if "nonzero_params" in summary.columns:
            summary["param_millions"] = summary["nonzero_params"]/1e6
        if "effective_param_bytes" in summary.columns:
            summary["effective_param_gb"] = summary["effective_param_bytes"]/1e9
        elif "storage_cost_bytes" in summary.columns:
            summary["effective_param_gb"] = summary["storage_cost_bytes"]/1e9
        return summary

    def compute_retrieval_degradation(self, domain: Optional[str] = None) -> pd.DataFrame:
        df = self.results_df
        if domain is not None and "eval_domain" in df.columns:
            df = df[df["eval_domain"] == domain]
        pivot = df.pivot_table(index="model_name", columns="prompt_len_theta", values="is_semantically_equivalent", aggfunc="mean")
        return pivot.round(4)

    def compute_domain_gap(self, theta: Optional[int] = None) -> pd.DataFrame:
        if theta is None:
            theta = max(self.config.theta_budgets)
        df = self.results_df
        df = df[df["prompt_len_theta"] == theta]
        if "eval_domain" not in df.columns:
            return pd.DataFrame()
        id_df = df[df["eval_domain"]=="id"].groupby("model_name").agg({"success_composite":"mean","semantic_similarity":"mean"}).rename(columns={"success_composite":"id_success","semantic_similarity":"id_sim"})
        ood_df = df[df["eval_domain"]=="ood"].groupby("model_name").agg({"success_composite":"mean","semantic_similarity":"mean"}).rename(columns={"success_composite":"ood_success","semantic_similarity":"ood_sim"})
        out = id_df.join(ood_df, how="outer")
        out["CGI_success"] = out["ood_success"]/out["id_success"]
        out["CGI_sim"] = out["ood_sim"]/out["id_sim"]
        return out

class FinetuneAnalyzer(ExperimentAnalyzer):
    def compute_model_efficiency(self, domain: Optional[str] = None, i_ref: Optional[float] = None) -> pd.DataFrame:
        summary = self.compute_summary_statistics(domain=domain)
        if i_ref is None:
            sentences = DataHandler.load_sentences(self.config.dataset_name, self.config.dataset_subset, self.config.test_split)
            all_text = " ".join(sentences)
            i_ref = len(all_text.encode('utf-8')) * 8  
        if "effective_param_gb" not in summary.columns:
            if "storage_cost_bytes" in summary.columns:
                summary["effective_param_gb"] = summary["storage_cost_bytes"]/1e9
            else:
                summary["effective_param_gb"] = np.nan
        summary["eir_success"] = (i_ref * summary["success_rate"]) / summary["storage_cost_bytes"]
        summary["eir_similarity"] = (i_ref * summary["avg_similarity"]) / summary["storage_cost_bytes"]
        if "param_millions" in summary.columns:
            summary["eir_success_per_paramM"] = (i_ref * summary["success_rate"]) / (summary["param_millions"] * 1e6)
            summary["eir_similarity_per_paramM"] = (i_ref * summary["avg_similarity"]) / (summary["param_millions"] * 1e6)
            summary["eir_successComp_per_paramM"] = (i_ref * summary["avg_success_composite"]) / (summary["param_millions"] * 1e6)
        return summary

class PruningAnalyzer(ExperimentAnalyzer):
    def compute_pruning_impact(self, domain: Optional[str] = None) -> pd.DataFrame:
        summary = self.compute_summary_statistics(domain=domain)
        if "sparsity" not in summary.columns:
            summary["sparsity"] = summary.index.to_series().str.extract(r"(\d+)%").astype(float).fillna(0)/100.0
        b = summary["sparsity"] == 0
        if b.any():
            baseline_perf = summary.loc[b,"success_rate"].iloc[0]
            summary["relative_performance"] = summary["success_rate"]/baseline_perf
            baseline_comp = summary.loc[b,"avg_success_composite"].iloc[0]
            summary["relative_comp_success"] = summary["avg_success_composite"]/baseline_comp
        else:
            summary["relative_performance"] = np.nan
            summary["relative_comp_success"] = np.nan
        return summary.sort_values("sparsity")

    def fit_scaling_law(self, df: pd.DataFrame, metric: str = "success_rate"):
        families = df['model_name'].apply(lambda x: x.split('-')[0] if '-' in x else x)  
        results = {}
        for family in families.unique():
            fam_df = df[families == family]
            if len(fam_df) < 3:
                continue  
            N = fam_df['nonzero_params'].values / fam_df['total_params'].values[0]  
            Q = fam_df[metric].values
            try:
                popt, _ = curve_fit(capacity_scaling_law, N, Q, p0=[1.0, 0.5, 0.5])
                results[family] = {'Q_max': popt[0], 'a': popt[1], 'gamma': popt[2]}
            except RuntimeError:
                results[family] = {'Q_max': np.nan, 'a': np.nan, 'gamma': np.nan}
        return results

    def compute_traditional_eir(self, sentences: List[str]) -> Dict[str, float]:
        traditional_eirs = {}

        # Huffman
        all_text = " ".join(sentences)
        huff_compressed, _ = huffman_compress(all_text)
        huff_size = len(huff_compressed.tobytes())
        i_ref = len(all_text.encode('utf-8')) * 8  
        traditional_eirs['Huffman'] = i_ref / huff_size  

        # LZW
        lzw_compressed = lzw_compress(all_text)
        lzw_size = len(lzw_compressed) * 12 / 8  # Approx bytes (12-bit codes)
        traditional_eirs['LZW'] = i_ref / lzw_size

        return traditional_eirs


