# 3_analyze_multi_model_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import bz2
import os
import yaml
from tqdm import tqdm
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
    'axes.grid': True,
    'grid.linewidth': 0.8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Define color palettes and styles for model families
FAMILY_COLORS = {
    'GPT': '#1f77b4',      # Blue
    'LLaMA': '#ff7f0e',    # Orange  
    'Cerebras': '#2ca02c', # Green
    'Mistral': '#d62728',  # Red
    'Qwen': '#9467bd',     # Purple
    'DeepSeek': '#8c564b', # Brown
    'Phi': '#e377c2',      # Pink
    'Gemma': '#7f7f7f',    # Gray
}

FAMILY_MARKERS = {
    'GPT': 'o',
    'LLaMA': 's', 
    'Cerebras': '^',
    'Mistral': 'D',
    'Qwen': 'v',
    'DeepSeek': 'p',
    'Phi': '*',
    'Gemma': 'h',
}

def extract_model_family(model_name):
    """Extract model family from model name"""
    model_name_lower = model_name.lower()
    if 'gpt' in model_name_lower:
        return 'GPT'
    elif 'llama' in model_name_lower:
        return 'LLaMA'
    elif 'cerebras' in model_name_lower:
        return 'Cerebras'
    elif 'mistral' in model_name_lower:
        return 'Mistral'
    elif 'qwen' in model_name_lower:
        return 'Qwen'
    elif 'deepseek' in model_name_lower:
        return 'DeepSeek'
    elif 'phi' in model_name_lower:
        return 'Phi'
    elif 'gemma' in model_name_lower:
        return 'Gemma'
    else:
        return 'Other'

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

def plot_retrieval_degradation_by_family(df, config, output_dir):
    """Create enhanced retrieval degradation plots grouped by model family"""
    
    theta_col = 'prompt_len_theta'
    
    # Add family information to dataframe
    df['model_family'] = df['model_name'].apply(extract_model_family)
    
    # Get retrieval data
    retrieval_df = df.groupby(['model_name', 'model_family', theta_col])['is_semantically_equivalent'].mean().reset_index()
    
    # Create the main plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: All families overlaid
    families = retrieval_df['model_family'].unique()
    
    for family in families:
        family_data = retrieval_df[retrieval_df['model_family'] == family]
        
        # Group by theta and get mean performance across models in family
        family_mean = family_data.groupby(theta_col)['is_semantically_equivalent'].agg(['mean', 'std']).reset_index()
        
        color = FAMILY_COLORS.get(family, '#000000')
        marker = FAMILY_MARKERS.get(family, 'o')
        
        # Plot mean line with error bars
        ax1.errorbar(family_mean[theta_col], family_mean['mean'], 
                    yerr=family_mean['std'], 
                    label=f'{family} Family', 
                    color=color, marker=marker, 
                    capsize=4, capthick=1.5, alpha=0.8)
        
        # Plot individual models as lighter lines
        for model in family_data['model_name'].unique():
            model_data = family_data[family_data['model_name'] == model]
            ax1.plot(model_data[theta_col], model_data['is_semantically_equivalent'], 
                    color=color, alpha=0.3, linewidth=1, linestyle='--')
    
    ax1.set_xlabel('Retrieval Budget θ (Prompt Length)', fontweight='bold')
    ax1.set_ylabel('Semantic Success Rate', fontweight='bold')
    ax1.set_title('Retrieval Performance Degradation by Model Family', fontweight='bold', pad=20)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Individual models with family grouping
    theta_values = sorted(retrieval_df[theta_col].unique())
    
    for i, family in enumerate(families):
        family_data = retrieval_df[retrieval_df['model_family'] == family]
        models_in_family = family_data['model_name'].unique()
        
        for j, model in enumerate(models_in_family):
            model_data = family_data[family_data['model_name'] == model]
            
            color = FAMILY_COLORS.get(family, '#000000')
            marker = FAMILY_MARKERS.get(family, 'o')
            
            # Add slight offset for visibility
            offset = (j - len(models_in_family)/2) * 0.02
            theta_offset = [t + offset for t in model_data[theta_col]]
            
            ax2.plot(theta_offset, model_data['is_semantically_equivalent'], 
                    label=model, color=color, marker=marker, alpha=0.7)
    
    ax2.set_xlabel('Retrieval Budget θ (Prompt Length)', fontweight='bold')
    ax2.set_ylabel('Semantic Success Rate', fontweight='bold')
    ax2.set_title('Individual Model Performance', fontweight='bold', pad=20)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "retrieval_degradation_by_family.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved enhanced retrieval degradation plot to {plot_path}")
    
    return plot_path

def plot_enhanced_scaling_analysis(summary_df, config, output_dir, bzip2_size):
    """Create enhanced scaling analysis visualization"""
    
    # Add family information
    summary_df['model_family'] = summary_df['Model (λ)'].apply(extract_model_family)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Storage Cost vs Performance by Family
    families = summary_df['model_family'].unique()
    for family in families:
        family_data = summary_df[summary_df['model_family'] == family]
        color = FAMILY_COLORS.get(family, '#000000')
        marker = FAMILY_MARKERS.get(family, 'o')
        
        ax1.scatter(family_data['Storage Cost (GB)'], family_data['Expected Fidelity'], 
                   label=family, color=color, marker=marker, s=100, alpha=0.8, edgecolors='black')
        
        # Add model name annotations
        for _, row in family_data.iterrows():
            ax1.annotate(row['Model (λ)'].split('-')[-1], 
                        (row['Storage Cost (GB)'], row['Expected Fidelity']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
    
    ax1.set_xlabel('Storage Cost (GB)', fontweight='bold')
    ax1.set_ylabel('Expected Fidelity', fontweight='bold')
    ax1.set_title('Storage Efficiency by Model Family', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: EIR Analysis
    EIR_SCALE_FACTOR = 1_000_000
    summary_df['EIR_scaled'] = (bzip2_size * summary_df['Expected Fidelity']) / (summary_df['Storage Cost (GB)'] * 1e9) * EIR_SCALE_FACTOR
    
    bars = ax2.bar(range(len(summary_df)), summary_df['EIR_scaled'], 
                   color=[FAMILY_COLORS.get(family, '#000000') for family in summary_df['model_family']],
                   alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Model', fontweight='bold')
    ax2.set_ylabel(f'EIR (×{EIR_SCALE_FACTOR:,})', fontweight='bold')
    ax2.set_title('Effective Information Ratio', fontweight='bold')
    ax2.set_xticks(range(len(summary_df)))
    ax2.set_xticklabels([name.split('-')[-1] for name in summary_df['Model (λ)']], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Success Rate vs Degradation Rate
    for family in families:
        family_data = summary_df[summary_df['model_family'] == family]
        color = FAMILY_COLORS.get(family, '#000000')
        marker = FAMILY_MARKERS.get(family, 'o')
        
        ax3.scatter(family_data['Success Rate (Semantic)'], family_data['Degradation Rate'], 
                   label=family, color=color, marker=marker, s=100, alpha=0.8, edgecolors='black')
    
    ax3.set_xlabel('Success Rate (Semantic)', fontweight='bold')
    ax3.set_ylabel('Degradation Rate', fontweight='bold')
    ax3.set_title('Performance vs Degradation Trade-off', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Factual Recall vs Fidelity
    for family in families:
        family_data = summary_df[summary_df['model_family'] == family]
        color = FAMILY_COLORS.get(family, '#000000')
        marker = FAMILY_MARKERS.get(family, 'o')
        
        ax4.scatter(family_data['Expected Fidelity'], family_data['Factual Recall'], 
                   label=family, color=color, marker=marker, s=100, alpha=0.8, edgecolors='black')
    
    ax4.set_xlabel('Expected Fidelity', fontweight='bold')
    ax4.set_ylabel('Factual Recall', fontweight='bold')
    ax4.set_title('Semantic vs Factual Performance', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "enhanced_scaling_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved enhanced scaling analysis to {plot_path}")
    
    return plot_path

def main():
    # This script is specifically for the multi-model fine-tuning experiment
    config_path = 'configs/finetune_experiment_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    results_path = f"./results/{config['experiment_name']}/raw_results.csv"
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}. Please run the corresponding experiment script first.")
        return
        
    df = pd.read_csv(results_path)
    df['is_semantically_equivalent'] = df['semantic_similarity'] >= config.get('semantic_threshold', 0.95)
    
    output_dir = f"./results/{config['experiment_name']}"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Advanced Metrics Analysis ---
    print("\n" + "="*80)
    print("--- ANALYSIS 1: ADVANCED METRICS FOR STORAGE DEGRADATION (Multi-Model) ---")
    
    theta_col = 'prompt_len_theta'
    theta_max = max(config['theta_budgets'])
    df_analysis = df[df[theta_col] == theta_max].copy()
    
    if NER_MODEL and 'original_sentence' in df_analysis.columns:
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
            'Storage Cost (GB)': group['storage_cost_lambda'].iloc[0] / 1e9,
            'Success Rate (Semantic)': group['is_semantically_equivalent'].mean(),
            'Degradation Rate': 1 - (group['is_semantically_equivalent'].mean() / success_rate_max),
            'Expected Fidelity': group['semantic_similarity'].mean(),
            'Factual Recall': group['intersection_count'].sum() / (group['ref_count'].sum() + 1e-9)
        })
    summary_df = pd.DataFrame(summary_data)
    # Reorder to match the order in the config file for logical progression
    model_order = [b['name'] for b in config['lambda_budgets']]
    summary_df['Model (λ)'] = pd.Categorical(summary_df['Model (λ)'], categories=model_order, ordered=True)
    summary_df = summary_df.sort_values('Model (λ)')
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # --- 2. Enhanced Retrieval Degradation Visualization ---
    print("\n" + "="*80)
    print("--- ANALYSIS 2: RETRIEVAL DEGRADATION (PERFORMANCE vs. THETA) ---")
    
    # Create enhanced visualization
    plot_retrieval_degradation_by_family(df, config, output_dir)
    
    retrieval_df = df.groupby(['model_name', theta_col])['is_semantically_equivalent'].mean().reset_index()
    retrieval_pivot = retrieval_df.pivot(index='model_name', columns=theta_col, values='is_semantically_equivalent')
    retrieval_pivot = retrieval_pivot.reindex(model_order)
    print("\nSemantic Success Rate at different Retrieval Budgets (θ):")
    print(retrieval_pivot.to_string(float_format="%.4f"))

    # --- 3. Enhanced EIR and Scaling Analysis ---
    print("\n" + "="*80)
    print("--- ANALYSIS 3: ENHANCED SCALING AND EIR ANALYSIS ---")
    all_text = "\n".join(df['original_sentence'].unique())
    bzip2_size = len(bz2.compress(all_text.encode('utf-8')))
    
    # Create enhanced scaling analysis visualization
    plot_enhanced_scaling_analysis(summary_df, config, output_dir, bzip2_size)
    
    eir_data = summary_df.copy()
    EIR_SCALE_FACTOR = 1_000_000
    eir_data['EIR_scaled'] = (bzip2_size * eir_data['Expected Fidelity']) / (eir_data['Storage Cost (GB)'] * 1e9) * EIR_SCALE_FACTOR
    
    print(f"\nEIR (x{EIR_SCALE_FACTOR:,}) Results:")
    print(eir_data[['Model (λ)', 'Expected Fidelity', 'EIR_scaled']].to_string(index=False, float_format="%.4f"))
    
    # --- 4. Scaling Law Analysis with FILTERING ---
    print("\n" + "="*80)
    print("--- ANALYSIS 4: SCALING LAWS (ACROSS ARCHITECTURES) ---")
    
    model_to_exclude = 'Cerebras-590M'
    print(f"NOTE: Fitting across different model families is illustrative, not scientifically rigorous.")
    print(f"Excluding '{model_to_exclude}' as an outlier for this illustrative fit.")
    scaling_df = summary_df[summary_df['Model (λ)'] != model_to_exclude]
    
    N_values = scaling_df['Storage Cost (GB)'].values * 1e9
    Q_values = scaling_df['Expected Fidelity'].values

    if len(N_values) < 3:
        print("  Not enough data points to fit scaling law after filtering.")
    else:
        try:
            popt, _ = curve_fit(capacity_scaling_law, N_values, Q_values, p0=[1.0, 1e10, 0.1], maxfev=8000)
            print(f"  Illustrative Capacity Scaling Law (γ) Fit Successful! γ = {popt[2]:.4f}, Q_max = {popt[0]:.4f}")

            plt.figure(figsize=(12, 8))
            # Plot points from the fit with family colors
            scaling_df_with_family = scaling_df.copy()
            scaling_df_with_family['model_family'] = scaling_df_with_family['Model (λ)'].apply(extract_model_family)
            
            for family in scaling_df_with_family['model_family'].unique():
                family_data = scaling_df_with_family[scaling_df_with_family['model_family'] == family]
                color = FAMILY_COLORS.get(family, '#000000')
                marker = FAMILY_MARKERS.get(family, 'o')
                
                plt.scatter(family_data['Storage Cost (GB)'] * 1e9, family_data['Expected Fidelity'], 
                           label=f'{family} Family', color=color, marker=marker, s=120, 
                           alpha=0.8, edgecolors='black', linewidth=1.5, zorder=5)
            
            # Plot the excluded point
            excluded_point = summary_df[summary_df['Model (λ)'] == model_to_exclude]
            if not excluded_point.empty:
                plt.scatter(excluded_point['Storage Cost (GB)'].iloc[0] * 1e9, excluded_point['Expected Fidelity'].iloc[0], 
                            label=f'Excluded: {model_to_exclude}', color='red', marker='x', s=150, linewidth=3)
            
            # Plot the fitted curve
            N_fine = np.linspace(min(N_values), max(N_values), 100)
            plt.plot(N_fine, capacity_scaling_law(N_fine, *popt), 
                    label=f'Illustrative Fit (γ={popt[2]:.3f})', color='black', linewidth=3, linestyle='--')
            
            plt.xlabel("Storage Budget N (Model Size in Bytes)", fontweight='bold')
            plt.ylabel("Performance Q (Expected Fidelity)", fontweight='bold')
            plt.title("Illustrative Scaling Law Fit Across Model Families", fontweight='bold', pad=20)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(output_dir, "enhanced_scaling_law_fit.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved enhanced scaling law plot to {plot_path}")

        except Exception as e:
            print(f"  Could not fit scaling law: {e}")

    print(f"\n✅ Analysis complete! All enhanced visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
