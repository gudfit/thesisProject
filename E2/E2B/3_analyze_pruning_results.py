# 3_analyze_pruning_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import bz2
import os
import torch
from transformers import AutoModelForCausalLM
from scipy.optimize import curve_fit

# --- Load NLP tools ---
try:
    NER_MODEL = spacy.load("en_core_web_sm")
    print("spaCy NER model loaded successfully.")
except IOError:
    print("Warning: spaCy model not found. Factual Recall will be skipped.")
    NER_MODEL = None

def get_entities(text: str) -> set:
    if not NER_MODEL or not isinstance(text, str) or not text.strip(): return set()
    doc = NER_MODEL(text)
    return {(ent.text.strip(), ent.label_) for ent in doc.ents}

def capacity_scaling_law(N, Q_max, a, gamma):
    return Q_max * (1 - a * np.power(N, -gamma))

def count_nonzero_params(model):
    return sum(torch.sum(p != 0).item() for p in model.parameters())

def main():
    BASE_MODEL_ID = "gpt2-medium"
    FINETUNED_PATH = f"./finetuned_models/{BASE_MODEL_ID.replace('/', '_')}"
    PRUNED_DIR = f"./pruned_models/{BASE_MODEL_ID.replace('/', '_')}"
    RESULTS_PATH = "./results/gpt2_medium_pruning_results.csv"

    if not os.path.exists(RESULTS_PATH):
        print(f"Error: Results file not found. Run 2_run_pruning_experiment.py first.")
        return

    df = pd.read_csv(RESULTS_PATH)
    df['is_semantically_equivalent'] = df['semantic_similarity'] >= 0.95
    
    # --- Step 1: Calculate TRUE Storage Cost (Non-zero params) ---
    print("Calculating true parameter counts for each pruned model...")
    param_counts = {}
    base_model_for_loading = AutoModelForCausalLM.from_pretrained(FINETUNED_PATH)
    pruned_files = sorted(os.listdir(PRUNED_DIR), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    for file_name in pruned_files:
        model_name = f"GPT2-medium ({file_name.split('.')[0].replace('_', ' ')}d)"
        model_path = os.path.join(PRUNED_DIR, file_name)
        base_model_for_loading.load_state_dict(torch.load(model_path))
        param_counts[model_name] = count_nonzero_params(base_model_for_loading)
    
    df['true_storage_cost_params'] = df['model_name'].map(param_counts)

    # --- Analysis 1: Advanced Metrics ---
    print("\n" + "="*80)
    print("--- ANALYSIS 1: ADVANCED METRICS FOR STORAGE DEGRADATION ---")
    theta_col = 'prompt_len_theta'
    theta_max = df[theta_col].max()
    df_analysis = df[df[theta_col] == theta_max].copy()
    
    if NER_MODEL:
        from tqdm import tqdm
        tqdm.pandas(desc="NER Analysis")
        df_analysis['ref_entities'] = df_analysis['original_sentence'].progress_apply(get_entities)
        df_analysis['gen_entities'] = df_analysis['reconstructed_sentence'].progress_apply(get_entities)
        df_analysis['intersection_count'] = df_analysis.apply(lambda r: len(r['ref_entities'].intersection(r['gen_entities'])), axis=1)
        df_analysis['ref_count'] = df_analysis['ref_entities'].apply(len)
    else:
        df_analysis['intersection_count'] = 0
        df_analysis['ref_count'] = 0

    best_model_row = df_analysis.loc[df_analysis['semantic_similarity'].idxmax()]
    success_rate_max = best_model_row['is_semantically_equivalent']
    
    summary_data = []
    for model_name, group in df_analysis.groupby('model_name'):
        summary_data.append({
            'Model (λ)': model_name,
            'Non-Zero Params (M)': group['true_storage_cost_params'].iloc[0] / 1e6,
            'Success Rate (Semantic)': group['is_semantically_equivalent'].mean(),
            'Degradation Rate': 1 - (group['is_semantically_equivalent'].mean() / success_rate_max),
            'Expected Fidelity': group['semantic_similarity'].mean(),
            'Factual Recall': group['intersection_count'].sum() / (group['ref_count'].sum() + 1e-9)
        })
    summary_df = pd.DataFrame(summary_data).sort_values(by='Non-Zero Params (M)', ascending=False)
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # --- Analysis 2: Retrieval Degradation ---
    print("\n" + "="*80)
    print("--- ANALYSIS 2: RETRIEVAL DEGRADATION (PERFORMANCE vs. THETA) ---")
    retrieval_df = df.groupby(['model_name', theta_col])['is_semantically_equivalent'].mean().reset_index()
    retrieval_pivot = retrieval_df.pivot(index='model_name', columns=theta_col, values='is_semantically_equivalent')
    # Reorder rows to match summary table
    retrieval_pivot = retrieval_pivot.reindex(summary_df['Model (λ)'].values)
    print("\nSemantic Success Rate at different Retrieval Budgets (θ):")
    print(retrieval_pivot.to_string(float_format="%.4f"))

    # --- Analysis 3: Effective Information Ratio (EIR) ---
    print("\n" + "="*80)
    print("--- ANALYSIS 3: EFFECTIVE INFORMATION RATIO (EIR) ---")
    all_text = "\n".join(df['original_sentence'].unique())
    bzip2_size = len(bz2.compress(all_text.encode('utf-8')))
    
    eir_data = summary_df.copy()
    EIR_SCALE_FACTOR = 1_000_000
    eir_data['EIR_scaled'] = (bzip2_size * eir_data['Expected Fidelity']) / (eir_data['Non-Zero Params (M)'] * 1e6) * EIR_SCALE_FACTOR
    
    print(f"\nEIR (x{EIR_SCALE_FACTOR:,}) Results:")
    print(eir_data[['Model (λ)', 'Expected Fidelity', 'EIR_scaled']].to_string(index=False, float_format="%.4f"))
    
    # --- Analysis 4: Scaling Laws ---
    print("\n" + "="*80)
    print("--- ANALYSIS 4: SCALING LAWS ---")
    
    scaling_df = summary_df.copy()
    N_values = scaling_df['Non-Zero Params (M)'].values * 1e6
    Q_values = scaling_df['Expected Fidelity'].values
    
    if len(N_values) < 3:
        print("  Not enough data points to fit scaling law.")
    else:
        try:
            popt, _ = curve_fit(capacity_scaling_law, N_values, Q_values, p0=[1.0, 1e9, 0.5], maxfev=8000, bounds=([0, 0, 0], [1.0, np.inf, np.inf]))
            print(f"  Capacity Scaling Law (γ) Fit Successful! γ = {popt[2]:.4f}, Q_max = {popt[0]:.4f}")

            plt.figure(figsize=(10, 6))
            plt.scatter(N_values, Q_values, label='Empirical Data (Pruned GPT2-Medium)', color='red', zorder=5)
            N_fine = np.linspace(min(N_values), max(N_values), 100)
            plt.plot(N_fine, capacity_scaling_law(N_fine, *popt), label=f'Fitted Law (γ={popt[2]:.3f})', color='blue')
            plt.xlabel("Storage Budget N (Number of Non-Zero Parameters)")
            plt.ylabel("Performance Q (Expected Fidelity)")
            plt.title("Capacity Scaling Law for Pruned GPT2-Medium")
            plt.legend(); plt.grid(True)
            plot_path = os.path.join("./results", "final_scaling_law_fit.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  Saved scaling law plot to {plot_path}")

        except Exception as e:
            print(f"  Could not fit scaling law: {e}")

if __name__ == "__main__":
    main()
