# 3_analyze_multi_model_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import bz2
import os
import yaml
from tqdm import tqdm
from scipy.optimize import curve_fit

try:
    NER_MODEL = spacy.load("en_core_web_sm")
    print("spaCy NER model loaded successfully.")
except IOError:
    print("Warning: spaCy model not found. Factual Recall will be skipped.")
    NER_MODEL = None


def get_entities(text: str) -> set:
    if not NER_MODEL or not isinstance(text, str) or not text.strip():
        return set()
    doc = NER_MODEL(text)
    return {(ent.text.strip(), ent.label_) for ent in doc.ents}


def capacity_scaling_law(N, Q_max, a, gamma):
    return Q_max * (1 - a * np.power(N, -gamma))


def main():
    config_path = "configs/finetune_experiment_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results_path = f"./results/{config['experiment_name']}/raw_results.csv"
    if not os.path.exists(results_path):
        print(
            f"Error: Results file not found at {results_path}. Please run the corresponding experiment script first."
        )
        return

    df = pd.read_csv(results_path)
    df["is_semantically_equivalent"] = df["semantic_similarity"] >= config.get(
        "semantic_threshold", 0.95
    )

    print("\n" + "=" * 80)
    print("--- ANALYSIS 1: ADVANCED METRICS FOR STORAGE DEGRADATION (Multi-Model) ---")

    theta_col = "prompt_len_theta"
    theta_max = max(config["theta_budgets"])
    df_analysis = df[df[theta_col] == theta_max].copy()

    if NER_MODEL and "original_sentence" in df_analysis.columns:
        tqdm.pandas(desc="NER Analysis")
        df_analysis["ref_entities"] = df_analysis["original_sentence"].progress_apply(
            get_entities
        )
        df_analysis["gen_entities"] = df_analysis[
            "reconstructed_sentence"
        ].progress_apply(get_entities)
        df_analysis["intersection_count"] = df_analysis.apply(
            lambda r: len(r["ref_entities"].intersection(r["gen_entities"])), axis=1
        )
        df_analysis["ref_count"] = df_analysis["ref_entities"].apply(len)
    else:
        df_analysis["intersection_count"] = 0
        df_analysis["ref_count"] = 0

    best_model_row = df_analysis.loc[df_analysis["semantic_similarity"].idxmax()]
    success_rate_max = best_model_row["is_semantically_equivalent"]

    summary_data = []
    for model_name, group in df_analysis.groupby("model_name"):
        summary_data.append(
            {
                "Model (λ)": model_name,
                "Storage Cost (GB)": group["storage_cost_lambda"].iloc[0] / 1e9,
                "Success Rate (Semantic)": group["is_semantically_equivalent"].mean(),
                "Degradation Rate": 1
                - (group["is_semantically_equivalent"].mean() / success_rate_max),
                "Expected Fidelity": group["semantic_similarity"].mean(),
                "Factual Recall": group["intersection_count"].sum()
                / (group["ref_count"].sum() + 1e-9),
            }
        )
    summary_df = pd.DataFrame(summary_data)
    model_order = [b["name"] for b in config["lambda_budgets"]]
    summary_df["Model (λ)"] = pd.Categorical(
        summary_df["Model (λ)"], categories=model_order, ordered=True
    )
    summary_df = summary_df.sort_values("Model (λ)")
    print(summary_df.to_string(index=False, float_format="%.4f"))

    print("\n" + "=" * 80)
    print("--- ANALYSIS 2: RETRIEVAL DEGRADATION (PERFORMANCE vs. THETA) ---")
    retrieval_df = (
        df.groupby(["model_name", theta_col])["is_semantically_equivalent"]
        .mean()
        .reset_index()
    )
    retrieval_pivot = retrieval_df.pivot(
        index="model_name", columns=theta_col, values="is_semantically_equivalent"
    )
    retrieval_pivot = retrieval_pivot.reindex(model_order)
    print("\nSemantic Success Rate at different Retrieval Budgets (θ):")
    print(retrieval_pivot.to_string(float_format="%.4f"))

    print("\n" + "=" * 80)
    print("--- ANALYSIS 3: EFFECTIVE INFORMATION RATIO (EIR) ---")
    all_text = "\n".join(df["original_sentence"].unique())
    bzip2_size = len(bz2.compress(all_text.encode("utf-8")))

    eir_data = summary_df.copy()
    EIR_SCALE_FACTOR = 1_000_000
    eir_data["EIR_scaled"] = (
        (bzip2_size * eir_data["Expected Fidelity"])
        / (eir_data["Storage Cost (GB)"] * 1e9)
        * EIR_SCALE_FACTOR
    )

    print(f"\nEIR (x{EIR_SCALE_FACTOR:,}) Results:")
    print(
        eir_data[["Model (λ)", "Expected Fidelity", "EIR_scaled"]].to_string(
            index=False, float_format="%.4f"
        )
    )

    print("\n" + "=" * 80)
    print("--- ANALYSIS 4: SCALING LAWS (ACROSS ARCHITECTURES) ---")

    model_to_exclude = "Cerebras-590M"
    print(
        f"NOTE: Fitting across different model families is illustrative, not scientifically rigorous."
    )
    print(f"Excluding '{model_to_exclude}' as an outlier for this illustrative fit.")
    scaling_df = summary_df[summary_df["Model (λ)"] != model_to_exclude]

    N_values = scaling_df["Storage Cost (GB)"].values * 1e9
    Q_values = scaling_df["Expected Fidelity"].values

    if len(N_values) < 3:
        print("  Not enough data points to fit scaling law after filtering.")
    else:
        try:
            popt, _ = curve_fit(
                capacity_scaling_law,
                N_values,
                Q_values,
                p0=[1.0, 1e10, 0.1],
                maxfev=8000,
            )
            print(
                f"  Illustrative Capacity Scaling Law (γ) Fit Successful! γ = {popt[2]:.4f}, Q_max = {popt[0]:.4f}"
            )

            plt.figure(figsize=(10, 6))
            plt.scatter(
                N_values, Q_values, label="Data used for fit", color="red", zorder=5
            )
            excluded_point = summary_df[summary_df["Model (λ)"] == model_to_exclude]
            if not excluded_point.empty:
                plt.scatter(
                    excluded_point["Storage Cost (GB)"].iloc[0] * 1e9,
                    excluded_point["Expected Fidelity"].iloc[0],
                    label=f"Excluded: {model_to_exclude}",
                    color="gray",
                    marker="x",
                    s=100,
                )
            N_fine = np.linspace(min(N_values), max(N_values), 100)
            plt.plot(
                N_fine,
                capacity_scaling_law(N_fine, *popt),
                label=f"Illustrative Fit (γ={popt[2]:.3f})",
                color="blue",
            )
            plt.xlabel("Storage Budget N (Model Size in Bytes)")
            plt.ylabel("Performance Q (Expected Fidelity)")
            plt.title("Illustrative Scaling Law Fit Across Model Families")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(
                f"./results/{config['experiment_name']}",
                "illustrative_scaling_law_fit.png",
            )
            plt.savefig(plot_path)
            plt.close()
            print(f"  Saved illustrative scaling law plot to {plot_path}")

        except Exception as e:
            print(f"  Could not fit scaling law: {e}")


if __name__ == "__main__":
    main()
