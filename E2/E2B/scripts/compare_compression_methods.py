#!/usr/bin/env python3
"""
Comprehensive analysis comparing all compression methods.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import bz2
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define color schemes for different methods
METHOD_COLORS = {
    'fine_tuning': '#1f77b4',
    'basic_pruning': '#ff7f0e', 
    'advanced_pruning': '#2ca02c',
    'quantization': '#d62728',
    'magnitude': '#9467bd',
    'structured': '#8c564b',
    'block_sparse': '#e377c2',
    'sparsegpt': '#7f7f7f'
}

class CompressionAnalyzer:
    """Comprehensive analyzer for compression method comparison."""
    
    def __init__(self, results_dirs: List[str], output_dir: str):
        self.results_dirs = results_dirs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.combined_df = None
        self.semantic_threshold = 0.95
        
    def load_and_label_results(self) -> pd.DataFrame:
        """Load results from multiple experiments and label them properly."""
        all_results = []
        
        for results_dir in self.results_dirs:
            results_path = Path(results_dir) / "raw_results.csv"
            if not results_path.exists():
                logger.warning(f"Results file not found: {results_path}")
                continue
                
            df = pd.read_csv(results_path)
            
            # Identify compression method from directory or data structure
            if 'quantization' in results_dir.lower():
                df['compression_method'] = 'quantization'
                # Extract quantization details if available
                if 'quantization_bits' in df.columns:
                    df['compression_details'] = df.apply(
                        lambda x: f"b{x['quantization_bits']}_s{int(x.get('sparsity', 0)*100)}", 
                        axis=1
                    )
            elif 'advanced_pruning' in results_dir.lower():
                df['compression_method'] = 'advanced_pruning'
                if 'pruning_method' in df.columns:
                    df['compression_details'] = df['pruning_method']
            elif 'pruning' in results_dir.lower():
                df['compression_method'] = 'basic_pruning'
                df['compression_details'] = 'magnitude'
            else:
                df['compression_method'] = 'fine_tuning'
                df['compression_details'] = 'baseline'
            
            # Ensure we have storage cost information
            if 'storage_cost_lambda' in df.columns and 'storage_cost_bytes' not in df.columns:
                df['storage_cost_bytes'] = df['storage_cost_lambda']
            elif 'storage_cost_bytes' not in df.columns:
                # Try to infer from model size
                df['storage_cost_bytes'] = df.groupby('model_name')['model_name'].transform(
                    lambda x: self._estimate_model_size(x.iloc[0])
                )
            
            # Ensure semantic equivalence column exists
            if 'is_semantically_equivalent' not in df.columns:
                df['is_semantically_equivalent'] = df['semantic_similarity'] >= self.semantic_threshold
            
            all_results.append(df)
        
        combined = pd.concat(all_results, ignore_index=True)
        
        # Fill NaN storage costs with estimates based on model name
        if combined['storage_cost_bytes'].isna().any():
            combined['storage_cost_bytes'] = combined.apply(
                lambda x: self._estimate_model_size(x['model_name']) 
                if pd.isna(x['storage_cost_bytes']) else x['storage_cost_bytes'], 
                axis=1
            )
        
        self.combined_df = combined
        return combined
    
    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size based on model name."""
        # Basic estimates in bytes
        size_map = {
            'gpt2': 5e8,  # ~500MB
            'gpt2-medium': 1.5e9,  # ~1.5GB
            'gpt2-large': 3e9,  # ~3GB
            'cerebras-111m': 4.5e8,  # ~450MB
            'cerebras-256m': 1e9,  # ~1GB
            'cerebras-590m': 2.4e9,  # ~2.4GB
        }
        
        model_lower = model_name.lower()
        for key, size in size_map.items():
            if key in model_lower:
                # Adjust for compression
                if 'pruned' in model_lower:
                    # Extract pruning percentage
                    import re
                    match = re.search(r'(\d+)(?:d|%)', model_lower)
                    if match:
                        prune_pct = int(match.group(1)) / 100
                        return size * (1 - prune_pct * 0.7)  # Rough estimate
                return size
        
        return 1e9  # Default 1GB
    
    def analyze_compression_method(self, method: str, method_df: pd.DataFrame):
        """Perform detailed analysis for a specific compression method."""
        print(f"\n{'='*80}")
        print(f"--- DETAILED ANALYSIS FOR {method.upper()} ---")
        print('='*80)
        
        # Get theta values
        theta_values = sorted(method_df['prompt_len_theta'].unique())
        theta_max = max(theta_values)
        
        # 1. Storage Degradation Analysis
        print("\n--- ANALYSIS 1: STORAGE DEGRADATION ---")
        df_max_theta = method_df[method_df['prompt_len_theta'] == theta_max]
        
        # Group by model and compression details
        if 'compression_details' in df_max_theta.columns:
            groupby_cols = ['model_name', 'compression_details']
        else:
            groupby_cols = ['model_name']
        
        summary_data = []
        for model_group, group_df in df_max_theta.groupby(groupby_cols):
            if isinstance(model_group, tuple):
                model_name = model_group[0]
                compression_detail = model_group[1] if len(model_group) > 1 else ''
            else:
                model_name = model_group
                compression_detail = ''
            
            # Calculate metrics
            success_rate = group_df['is_semantically_equivalent'].mean()
            
            # Get best model for degradation calculation
            best_success_rate = df_max_theta['is_semantically_equivalent'].max()
            degradation = 1 - (success_rate / best_success_rate) if best_success_rate > 0 else 0
            
            summary_data.append({
                'Model (λ)': f"{model_name} ({compression_detail})" if compression_detail else model_name,
                'Storage (GB)': group_df['storage_cost_bytes'].iloc[0] / 1e9,
                'Non-Zero Params (M)': group_df.get('nonzero_params', group_df['storage_cost_bytes']).iloc[0] / 1e6,
                'Success Rate (Semantic)': success_rate,
                'Degradation Rate': degradation,
                'Expected Fidelity': group_df['semantic_similarity'].mean(),
                'Avg Latency (ms)': group_df['retrieval_cost_ms'].mean()
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False, float_format="%.4f"))
        
        # 2. Retrieval Degradation Analysis
        print(f"\n--- ANALYSIS 2: RETRIEVAL DEGRADATION (PERFORMANCE vs. THETA) ---")
        
        # Create pivot table for retrieval performance
        retrieval_pivot = method_df.pivot_table(
            index='model_name',
            columns='prompt_len_theta',
            values='is_semantically_equivalent',
            aggfunc='mean'
        )
        
        print("\nSemantic Success Rate at different Retrieval Budgets (θ):")
        print(retrieval_pivot.to_string(float_format="%.4f"))
        
        # 3. EIR Analysis
        print(f"\n--- ANALYSIS 3: EFFECTIVE INFORMATION RATIO (EIR) ---")
        
        # Calculate EIR using compressed text size
        unique_sentences = method_df['original_sentence'].unique()
        all_text = "\n".join(unique_sentences)
        bzip2_size = len(bz2.compress(all_text.encode('utf-8')))
        
        EIR_SCALE_FACTOR = 1_000_000
        eir_data = summary_df.copy()
        eir_data['EIR_scaled'] = (
            (bzip2_size * eir_data['Expected Fidelity']) / 
            (eir_data['Storage (GB)'] * 1e9) * EIR_SCALE_FACTOR
        )
        
        print(f"\nEIR (x{EIR_SCALE_FACTOR:,}) Results:")
        print(eir_data[['Model (λ)', 'Expected Fidelity', 'EIR_scaled']].to_string(
            index=False, float_format="%.4f"
        ))
        
        # 4. Method-specific insights
        if method == 'quantization':
            self._analyze_quantization_specific(method_df)
        elif 'pruning' in method:
            self._analyze_pruning_specific(method_df)
        
        return summary_df, retrieval_pivot, eir_data
    
    def _analyze_quantization_specific(self, df):
        """Analyze quantization-specific metrics."""
        print("\n--- QUANTIZATION-SPECIFIC ANALYSIS ---")
        
        if 'quantization_bits' in df.columns and 'sparsity' in df.columns:
            quant_summary = df.groupby(['quantization_bits', 'sparsity']).agg({
                'semantic_similarity': 'mean',
                'retrieval_cost_ms': 'mean',
                'storage_cost_bytes': 'mean'
            }).round(4)
            
            print("\nPerformance by Quantization Configuration:")
            print(quant_summary.to_string())
    
    def _analyze_pruning_specific(self, df):
        """Analyze pruning-specific metrics."""
        print("\n--- PRUNING-SPECIFIC ANALYSIS ---")
        
        if 'pruning_amount' in df.columns:
            # Analyze performance vs pruning amount
            prune_summary = df.groupby('pruning_amount').agg({
                'semantic_similarity': 'mean',
                'retrieval_cost_ms': 'mean',
                'nonzero_params': 'mean'
            }).round(4)
            
            print("\nPerformance by Pruning Amount:")
            print(prune_summary.to_string())
    
    def create_comprehensive_visualizations(self, method_summaries: Dict):
        """Create comprehensive visualization suite."""
        print("\n--- CREATING COMPREHENSIVE VISUALIZATIONS ---")
        
        # 1. Multi-method comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Performance vs Storage
        for method, (summary_df, _, _) in method_summaries.items():
            color = METHOD_COLORS.get(method, '#000000')
            ax1.scatter(summary_df['Storage (GB)'], 
                       summary_df['Success Rate (Semantic)'],
                       label=method, color=color, s=100, alpha=0.7, edgecolors='black')
        
        ax1.set_xlabel('Storage Cost (GB)', fontweight='bold')
        ax1.set_ylabel('Semantic Success Rate', fontweight='bold')
        ax1.set_title('Performance vs Storage by Compression Method', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Latency comparison
        latency_data = []
        for method, (summary_df, _, _) in method_summaries.items():
            for _, row in summary_df.iterrows():
                latency_data.append({
                    'Method': method,
                    'Model': row['Model (λ)'],
                    'Latency': row['Avg Latency (ms)']
                })
        
        latency_df = pd.DataFrame(latency_data)
        sns.boxplot(data=latency_df, x='Method', y='Latency', ax=ax2)
        ax2.set_ylabel('Retrieval Latency (ms)', fontweight='bold')
        ax2.set_title('Latency Distribution by Compression Method', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # EIR comparison
        for method, (_, _, eir_data) in method_summaries.items():
            color = METHOD_COLORS.get(method, '#000000')
            ax3.bar(range(len(eir_data)), eir_data['EIR_scaled'], 
                   label=method, color=color, alpha=0.7)
        
        ax3.set_ylabel('EIR (×1,000,000)', fontweight='bold')
        ax3.set_title('Effective Information Ratio Comparison', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Degradation patterns
        for method, (_, retrieval_pivot, _) in method_summaries.items():
            if not retrieval_pivot.empty:
                mean_degradation = retrieval_pivot.mean(axis=0)
                color = METHOD_COLORS.get(method, '#000000')
                ax4.plot(mean_degradation.index, mean_degradation.values,
                        label=method, color=color, marker='o', linewidth=2)
        
        ax4.set_xlabel('Retrieval Budget θ', fontweight='bold')
        ax4.set_ylabel('Average Success Rate', fontweight='bold')
        ax4.set_title('Retrieval Degradation Patterns', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Create heatmap of all methods and models
        self._create_performance_heatmap()
    
    def _create_performance_heatmap(self):
        """Create a heatmap showing performance across all methods and models."""
        # Prepare data for heatmap
        heatmap_data = self.combined_df.pivot_table(
            index='model_name',
            columns='compression_method',
            values='semantic_similarity',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   cbar_kws={'label': 'Average Semantic Similarity'},
                   vmin=0, vmax=1)
        plt.title('Performance Heatmap: Models × Compression Methods', fontweight='bold', pad=20)
        plt.xlabel('Compression Method', fontweight='bold')
        plt.ylabel('Model', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def fit_scaling_laws(self, method_summaries: Dict):
        """Fit and compare scaling laws across methods."""
        print("\n--- SCALING LAW ANALYSIS ACROSS METHODS ---")
        
        def capacity_scaling_law(N, Q_max, a, gamma):
            return Q_max * (1 - a * np.power(N, -gamma))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for method, (summary_df, _, _) in method_summaries.items():
            if len(summary_df) < 3:
                print(f"Insufficient data for scaling law fit in {method}")
                continue
            
            N_values = summary_df['Storage (GB)'].values * 1e9
            Q_values = summary_df['Expected Fidelity'].values
            
            try:
                popt, _ = curve_fit(capacity_scaling_law, N_values, Q_values, 
                                   p0=[1.0, 1e9, 0.5], maxfev=8000)
                
                print(f"\n{method.upper()} Scaling Law:")
                print(f"  γ = {popt[2]:.4f}, Q_max = {popt[0]:.4f}")
                
                # Plot data and fit
                color = METHOD_COLORS.get(method, '#000000')
                ax.scatter(N_values, Q_values, label=f'{method} (data)', 
                          color=color, s=100, alpha=0.7)
                
                N_fine = np.linspace(min(N_values), max(N_values), 100)
                ax.plot(N_fine, capacity_scaling_law(N_fine, *popt),
                       label=f'{method} (γ={popt[2]:.3f})', 
                       color=color, linewidth=2, linestyle='--')
                
            except Exception as e:
                print(f"Could not fit scaling law for {method}: {e}")
        
        ax.set_xlabel('Storage Budget N (Bytes)', fontweight='bold')
        ax.set_ylabel('Performance Q (Expected Fidelity)', fontweight='bold')
        ax.set_title('Scaling Laws Comparison Across Compression Methods', fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scaling_laws_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, method_summaries: Dict):
        """Generate a comprehensive summary report."""
        report_path = self.output_dir / 'compression_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE COMPRESSION METHOD ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            
            overall_stats = self.combined_df.groupby('compression_method').agg({
                'semantic_similarity': ['mean', 'std', 'max'],
                'retrieval_cost_ms': ['mean', 'std', 'min'],
                'storage_cost_bytes': 'mean',
                'is_semantically_equivalent': 'mean'
            }).round(4)
            
            f.write(overall_stats.to_string() + "\n\n")
            
            # Best configurations
            f.write("BEST CONFIGURATIONS BY METRIC\n")
            f.write("-" * 40 + "\n")
            
            # Best for semantic similarity
            best_semantic = self.combined_df.loc[self.combined_df['semantic_similarity'].idxmax()]
            f.write(f"Best Semantic Similarity: {best_semantic['model_name']} "
                   f"({best_semantic['compression_method']}) - {best_semantic['semantic_similarity']:.4f}\n")
            
            # Best for speed
            best_speed = self.combined_df.loc[self.combined_df['retrieval_cost_ms'].idxmin()]
            f.write(f"Fastest Retrieval: {best_speed['model_name']} "
                   f"({best_speed['compression_method']}) - {best_speed['retrieval_cost_ms']:.2f}ms\n")
            
            # Best for storage
            best_storage = self.combined_df.loc[self.combined_df['storage_cost_bytes'].idxmin()]
            f.write(f"Smallest Storage: {best_storage['model_name']} "
                   f"({best_storage['compression_method']}) - {best_storage['storage_cost_bytes']/1e9:.3f}GB\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Analysis complete. See visualizations for detailed comparisons.\n")
        
        print(f"\nSummary report saved to {report_path}")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("COMPREHENSIVE COMPRESSION METHOD ANALYSIS")
        print("=" * 80)
        
        # Load all data
        print("\nLoading results from all experiments...")
        self.load_and_label_results()
        
        # Save combined results
        self.combined_df.to_csv(self.output_dir / 'combined_results.csv', index=False)
        print(f"Combined results saved to {self.output_dir / 'combined_results.csv'}")
        
        # Analyze each compression method
        method_summaries = {}
        for method in self.combined_df['compression_method'].unique():
            method_df = self.combined_df[self.combined_df['compression_method'] == method]
            summaries = self.analyze_compression_method(method, method_df)
            method_summaries[method] = summaries
        
        # Create visualizations
        self.create_comprehensive_visualizations(method_summaries)
        
        # Fit scaling laws
        self.fit_scaling_laws(method_summaries)
        
        # Generate summary report
        self.generate_summary_report(method_summaries)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"All results saved to: {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis of LLM compression methods"
    )
    parser.add_argument(
        '--results-dirs', 
        nargs='+', 
        required=True,
        help='Directories containing experiment results'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='results/comprehensive_analysis',
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--semantic-threshold',
        type=float,
        default=0.95,
        help='Threshold for semantic equivalence'
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = CompressionAnalyzer(args.results_dirs, args.output_dir)
    analyzer.semantic_threshold = args.semantic_threshold
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
