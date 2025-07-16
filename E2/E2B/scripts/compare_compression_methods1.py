#!/usr/bin/env python3
"""
Fixed comprehensive analysis that respects scientific validity.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import bz2
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define color schemes
METHOD_COLORS = {
    'fine_tuning': '#1f77b4',
    'basic_pruning': '#ff7f0e', 
    'advanced_pruning': '#2ca02c',
    'quantization': '#d62728',
}

MODEL_FAMILY_COLORS = {
    'gpt2': '#1f77b4',
    'cerebras': '#2ca02c',
    'llama': '#ff7f0e',
    'mistral': '#d62728',
}

class ScientificallyValidAnalyzer:
    """Analyzer that respects the scientific validity of comparisons."""
    
    def __init__(self, results_dirs: List[str], output_dir: str):
        self.results_dirs = results_dirs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.combined_df = None
        self.semantic_threshold = 0.95
        
    def load_and_label_results(self) -> pd.DataFrame:
        """Load results and properly label them."""
        all_results = []
        
        for results_dir in self.results_dirs:
            results_path = Path(results_dir) / "raw_results.csv"
            if not results_path.exists():
                logger.warning(f"Results file not found: {results_path}")
                continue
                
            df = pd.read_csv(results_path)
            
            # Identify experiment type and extract base model
            exp_name = Path(results_dir).name
            
            # Label compression method
            if 'quantization' in exp_name.lower():
                df['compression_method'] = 'quantization'
            elif 'advanced_pruning' in exp_name.lower():
                df['compression_method'] = 'advanced_pruning'
            elif 'pruning' in exp_name.lower():
                df['compression_method'] = 'basic_pruning'
            else:
                df['compression_method'] = 'fine_tuning'
            
            # Extract base model architecture
            df['base_architecture'] = df['model_name'].apply(self._extract_base_architecture)
            
            # Ensure storage cost exists
            if 'storage_cost_bytes' not in df.columns:
                if 'storage_cost_lambda' in df.columns:
                    df['storage_cost_bytes'] = df['storage_cost_lambda']
                else:
                    df['storage_cost_bytes'] = df['model_name'].apply(self._estimate_model_size)
            
            # Extract compression level for pruning
            if 'pruning' in df['compression_method'].iloc[0]:
                df['compression_level'] = df['model_name'].apply(self._extract_pruning_level)
            elif df['compression_method'].iloc[0] == 'quantization':
                # Extract bits and sparsity
                if 'quantization_bits' in df.columns:
                    df['compression_level'] = df.apply(
                        lambda x: f"b{x.get('quantization_bits', 8)}_s{int(x.get('sparsity', 0)*100)}", 
                        axis=1
                    )
                else:
                    df['compression_level'] = 'unknown'
            else:
                df['compression_level'] = 'baseline'
            
            # Ensure semantic equivalence
            if 'is_semantically_equivalent' not in df.columns:
                df['is_semantically_equivalent'] = df['semantic_similarity'] >= self.semantic_threshold
            
            all_results.append(df)
        
        self.combined_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        return self.combined_df
    
    def _extract_base_architecture(self, model_name: str) -> str:
        """Extract the base architecture from model name."""
        model_lower = model_name.lower()
        
        # Common patterns
        architectures = {
            'gpt2-medium': 'GPT2-Medium',
            'gpt2-small': 'GPT2-Small',
            'gpt2': 'GPT2-Small',
            'cerebras-111m': 'Cerebras-111M',
            'cerebras-256m': 'Cerebras-256M',
            'cerebras-590m': 'Cerebras-590M',
            'llama': 'LLaMA',
            'mistral': 'Mistral',
        }
        
        for pattern, arch in architectures.items():
            if pattern in model_lower:
                return arch
        
        # Try to extract from name
        parts = model_name.split('_')
        if parts:
            return parts[0]
        
        return 'Unknown'
    
    def _extract_pruning_level(self, model_name: str) -> float:
        """Extract pruning percentage from model name."""
        match = re.search(r'pruned[_\s]+(\d+)', model_name)
        if match:
            return float(match.group(1)) / 100
        return 0.0
    
    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size based on architecture."""
        size_map = {
            'gpt2': 5e8,
            'gpt2-medium': 1.5e9,
            'gpt2-large': 3e9,
            'cerebras-111m': 4.5e8,
            'cerebras-256m': 1e9,
            'cerebras-590m': 2.4e9,
        }
        
        model_lower = model_name.lower()
        
        # Find base size
        base_size = 1e9  # default
        for key, size in size_map.items():
            if key in model_lower:
                base_size = size
                break
        
        # Adjust for pruning
        if 'pruned' in model_lower:
            prune_level = self._extract_pruning_level(model_name)
            # Rough estimate: pruning reduces size by ~70% of the pruning percentage
            return base_size * (1 - prune_level * 0.7)
        
        return base_size
    
    def analyze_by_architecture_and_method(self):
        """Analyze results grouped by architecture and compression method."""
        print("\n" + "="*80)
        print("ANALYSIS BY ARCHITECTURE AND COMPRESSION METHOD")
        print("="*80)
        
        # Group by architecture
        architectures = self.combined_df['base_architecture'].unique()
        
        for arch in sorted(architectures):
            arch_df = self.combined_df[self.combined_df['base_architecture'] == arch]
            methods = arch_df['compression_method'].unique()
            
            if len(methods) > 1:  # Only analyze if multiple methods exist
                print(f"\n{'='*60}")
                print(f"ARCHITECTURE: {arch}")
                print('='*60)
                
                for method in sorted(methods):
                    method_df = arch_df[arch_df['compression_method'] == method]
                    self._analyze_single_architecture_method(arch, method, method_df)
    
    def _analyze_single_architecture_method(self, arch: str, method: str, df: pd.DataFrame):
        """Analyze a single architecture-method combination."""
        print(f"\n--- {method.upper()} on {arch} ---")
        
        theta_max = df['prompt_len_theta'].max()
        df_max_theta = df[df['prompt_len_theta'] == theta_max]
        
        # For pruning/quantization, show progression
        if method in ['basic_pruning', 'advanced_pruning', 'quantization']:
            # Group by compression level
            summary_data = []
            
            for level, group in df_max_theta.groupby('compression_level'):
                # Calculate metrics
                storage_gb = group['storage_cost_bytes'].iloc[0] / 1e9
                
                # For pruning, calculate non-zero params
                if 'nonzero_params' in group.columns:
                    nonzero_m = group['nonzero_params'].iloc[0] / 1e6
                else:
                    # Estimate from pruning level
                    if method == 'basic_pruning' and isinstance(level, float):
                        base_params = storage_gb * 1e9 / 4  # Rough estimate: 4 bytes per param
                        nonzero_m = base_params * (1 - level) / 1e6
                    else:
                        nonzero_m = storage_gb * 250  # Very rough estimate
                
                summary_data.append({
                    'Config': f"{arch} ({method} {level})",
                    'Storage (GB)': storage_gb,
                    'Non-Zero Params (M)': nonzero_m,
                    'Success Rate': group['is_semantically_equivalent'].mean(),
                    'Expected Fidelity': group['semantic_similarity'].mean(),
                    'Avg Latency (ms)': group['retrieval_cost_ms'].mean()
                })
            
            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False, float_format="%.4f"))
            
            # Retrieval degradation
            print(f"\nRetrieval Degradation for {method} on {arch}:")
            pivot = df.pivot_table(
                index='compression_level',
                columns='prompt_len_theta',
                values='is_semantically_equivalent',
                aggfunc='mean'
            )
            print(pivot.to_string(float_format="%.4f"))
            
            # EIR Analysis
            if len(summary_df) > 0:
                self._calculate_eir(summary_df, df)
            
            # Scaling law - ONLY if same architecture with multiple compression levels
            if len(summary_df) >= 4 and method in ['basic_pruning', 'advanced_pruning']:
                self._fit_architecture_specific_scaling(arch, method, summary_df)
    
    def _calculate_eir(self, summary_df: pd.DataFrame, full_df: pd.DataFrame):
        """Calculate EIR for a specific configuration."""
        print("\nEffective Information Ratio (EIR):")
        
        # Get unique sentences for compression
        unique_sentences = full_df['original_sentence'].unique()
        all_text = "\n".join(unique_sentences)
        bzip2_size = len(bz2.compress(all_text.encode('utf-8')))
        
        EIR_SCALE_FACTOR = 1_000_000
        summary_df['EIR_scaled'] = (
            (bzip2_size * summary_df['Expected Fidelity']) / 
            (summary_df['Storage (GB)'] * 1e9) * EIR_SCALE_FACTOR
        )
        
        print(summary_df[['Config', 'Expected Fidelity', 'EIR_scaled']].to_string(
            index=False, float_format="%.4f"
        ))
    
    def _fit_architecture_specific_scaling(self, arch: str, method: str, summary_df: pd.DataFrame):
        """Fit scaling law for a specific architecture and method."""
        print(f"\nScaling Law Analysis for {arch} with {method}:")
        
        def capacity_scaling_law(N, Q_max, a, gamma):
            return Q_max * (1 - a * np.power(N, -gamma))
        
        # Use non-zero params or storage as N
        if 'Non-Zero Params (M)' in summary_df.columns:
            N_values = summary_df['Non-Zero Params (M)'].values * 1e6
        else:
            N_values = summary_df['Storage (GB)'].values * 1e9
        
        Q_values = summary_df['Expected Fidelity'].values
        
        # Only fit if we have enough variation
        if len(N_values) >= 4 and N_values.std() > 0:
            try:
                popt, _ = curve_fit(
                    capacity_scaling_law, N_values, Q_values, 
                    p0=[1.0, np.median(N_values), 0.5], 
                    maxfev=8000,
                    bounds=([0, 0, 0], [1.0, np.inf, np.inf])
                )
                
                print(f"  Scaling Law Fit: γ = {popt[2]:.4f}, Q_max = {popt[0]:.4f}")
                
                # Create plot for this specific architecture
                plt.figure(figsize=(10, 6))
                plt.scatter(N_values, Q_values, label='Data', color='red', s=100)
                
                N_fine = np.linspace(min(N_values), max(N_values), 100)
                plt.plot(N_fine, capacity_scaling_law(N_fine, *popt), 
                        label=f'Fit (γ={popt[2]:.3f})', color='blue', linewidth=2)
                
                plt.xlabel('Non-Zero Parameters' if 'Non-Zero Params (M)' in summary_df.columns else 'Storage (Bytes)')
                plt.ylabel('Expected Fidelity')
                plt.title(f'Scaling Law: {arch} with {method}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                save_path = self.output_dir / f'scaling_{arch}_{method}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  Scaling plot saved to {save_path}")
                
            except Exception as e:
                print(f"  Could not fit scaling law: {e}")
        else:
            print("  Insufficient data variation for scaling law fit")
    
    def create_method_comparison_plots(self):
        """Create plots comparing methods within each architecture."""
        print("\n--- Creating Comparison Visualizations ---")
        
        architectures = self.combined_df['base_architecture'].unique()
        
        for arch in architectures:
            arch_df = self.combined_df[self.combined_df['base_architecture'] == arch]
            methods = arch_df['compression_method'].unique()
            
            if len(methods) <= 1:
                continue
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Compression Method Comparison: {arch}', fontsize=16, fontweight='bold')
            
            # 1. Performance vs Storage
            for method in methods:
                method_df = arch_df[arch_df['compression_method'] == method]
                theta_max = method_df['prompt_len_theta'].max()
                perf_df = method_df[method_df['prompt_len_theta'] == theta_max]
                
                # Group by model to get unique points
                grouped = perf_df.groupby('model_name').agg({
                    'storage_cost_bytes': 'first',
                    'semantic_similarity': 'mean',
                    'is_semantically_equivalent': 'mean'
                }).reset_index()
                
                color = METHOD_COLORS.get(method, '#000000')
                ax1.scatter(grouped['storage_cost_bytes'] / 1e9, 
                           grouped['semantic_similarity'],
                           label=method, color=color, s=100, alpha=0.7)
            
            ax1.set_xlabel('Storage (GB)')
            ax1.set_ylabel('Semantic Similarity')
            ax1.set_title('Performance vs Storage')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Retrieval speed by theta
            for method in methods:
                method_df = arch_df[arch_df['compression_method'] == method]
                speed_by_theta = method_df.groupby('prompt_len_theta')['retrieval_cost_ms'].mean()
                
                color = METHOD_COLORS.get(method, '#000000')
                ax2.plot(speed_by_theta.index, speed_by_theta.values,
                        label=method, color=color, marker='o', linewidth=2)
            
            ax2.set_xlabel('Prompt Length (θ)')
            ax2.set_ylabel('Avg Retrieval Time (ms)')
            ax2.set_title('Retrieval Speed')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Success rate by theta
            for method in methods:
                method_df = arch_df[arch_df['compression_method'] == method]
                success_by_theta = method_df.groupby('prompt_len_theta')['is_semantically_equivalent'].mean()
                
                color = METHOD_COLORS.get(method, '#000000')
                ax3.plot(success_by_theta.index, success_by_theta.values,
                        label=method, color=color, marker='s', linewidth=2)
            
            ax3.set_xlabel('Prompt Length (θ)')
            ax3.set_ylabel('Semantic Success Rate')
            ax3.set_title('Success Rate Degradation')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.05)
            
            # 4. Method comparison bar chart
            method_summary = []
            for method in methods:
                method_df = arch_df[arch_df['compression_method'] == method]
                method_summary.append({
                    'Method': method,
                    'Avg Similarity': method_df['semantic_similarity'].mean(),
                    'Avg Success': method_df['is_semantically_equivalent'].mean(),
                    'Avg Speed': method_df['retrieval_cost_ms'].mean()
                })
            
            method_summary_df = pd.DataFrame(method_summary)
            x = range(len(method_summary_df))
            width = 0.25
            
            ax4.bar([i - width for i in x], method_summary_df['Avg Similarity'], 
                   width, label='Similarity', alpha=0.8)
            ax4.bar(x, method_summary_df['Avg Success'], 
                   width, label='Success Rate', alpha=0.8)
            ax4.bar([i + width for i in x], method_summary_df['Avg Speed'] / 1000, 
                   width, label='Speed (s)', alpha=0.8)
            
            ax4.set_xlabel('Compression Method')
            ax4.set_ylabel('Value')
            ax4.set_title('Method Summary')
            ax4.set_xticks(x)
            ax4.set_xticklabels(method_summary_df['Method'], rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            save_path = self.output_dir / f'comparison_{arch.replace(" ", "_")}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved comparison plot for {arch} to {save_path}")
    
    def generate_summary_report(self):
        """Generate a scientifically valid summary report."""
        report_path = self.output_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("COMPRESSION METHOD ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("KEY FINDINGS\n")
            f.write("-"*40 + "\n\n")
            
            # Best configurations per architecture
            f.write("BEST CONFIGURATIONS BY ARCHITECTURE:\n\n")
            
            for arch in self.combined_df['base_architecture'].unique():
                arch_df = self.combined_df[self.combined_df['base_architecture'] == arch]
                
                # Best semantic similarity
                best_sim = arch_df.loc[arch_df['semantic_similarity'].idxmax()]
                f.write(f"{arch}:\n")
                f.write(f"  Best Similarity: {best_sim['model_name']} "
                       f"({best_sim['compression_method']}) - {best_sim['semantic_similarity']:.4f}\n")
                
                # Most efficient (best performance per GB)
                arch_df['efficiency'] = arch_df['semantic_similarity'] / (arch_df['storage_cost_bytes'] / 1e9)
                best_eff = arch_df.loc[arch_df['efficiency'].idxmax()]
                f.write(f"  Most Efficient: {best_eff['model_name']} "
                       f"({best_eff['compression_method']}) - {best_eff['efficiency']:.2f} sim/GB\n\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Note: Scaling laws are only fitted within the same architecture\n")
            f.write("      with varying compression levels (scientifically valid).\n")
        
        print(f"\nSummary report saved to {report_path}")
    
    def run_analysis(self):
        """Run the complete scientifically valid analysis."""
        print("SCIENTIFICALLY VALID COMPRESSION ANALYSIS")
        print("="*80)
        
        # Load data
        print("\nLoading results...")
        self.load_and_label_results()
        
        if self.combined_df.empty:
            print("No data loaded! Check your results directories.")
            return
        
        # Save combined results
        self.combined_df.to_csv(self.output_dir / 'combined_results.csv', index=False)
        print(f"Combined results saved to {self.output_dir / 'combined_results.csv'}")
        
        # Analyze by architecture and method
        self.analyze_by_architecture_and_method()
        
        # Create comparison plots
        self.create_method_comparison_plots()
        
        # Generate report
        self.generate_summary_report()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Scientifically valid analysis of compression methods"
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
        default='results/scientific_analysis',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    analyzer = ScientificallyValidAnalyzer(args.results_dirs, args.output_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
