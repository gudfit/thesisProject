#!/usr/bin/env python3
"""
Compare results across different compression methods.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_and_label_results(results_dirs):
    """Load results from multiple experiments and label them."""
    all_results = []
    
    for results_dir in results_dirs:
        results_path = Path(results_dir) / "raw_results.csv"
        if not results_path.exists():
            print(f"Warning: {results_path} not found")
            continue
            
        df = pd.read_csv(results_path)
        
        # Add compression method label based on directory name
        if 'quantization' in results_dir:
            df['compression_method'] = 'quantization'
        elif 'advanced_pruning' in results_dir:
            df['compression_method'] = 'advanced_pruning'
        elif 'pruning' in results_dir:
            df['compression_method'] = 'basic_pruning'
        else:
            df['compression_method'] = 'fine_tuning'
            
        all_results.append(df)
    
    return pd.concat(all_results, ignore_index=True)

def create_comparison_plots(df, output_dir):
    """Create comparison visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance vs Compression Rate
    plt.figure(figsize=(12, 8))
    
    # Calculate compression rate
    if 'nonzero_params' in df.columns:
        # Group by model and calculate average metrics
        summary = df.groupby(['model_name', 'compression_method']).agg({
            'semantic_similarity': 'mean',
            'storage_cost_bytes': 'first',
            'nonzero_params': 'first'
        }).reset_index()
        
        # Plot
        for method in summary['compression_method'].unique():
            method_data = summary[summary['compression_method'] == method]
            plt.scatter(method_data['storage_cost_bytes'] / 1e9, 
                       method_data['semantic_similarity'],
                       label=method, s=100, alpha=0.7)
    
    plt.xlabel('Storage Cost (GB)')
    plt.ylabel('Average Semantic Similarity')
    plt.title('Compression Method Comparison: Performance vs Storage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'compression_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Retrieval Speed Comparison
    plt.figure(figsize=(12, 8))
    
    speed_summary = df.groupby(['compression_method', 'prompt_len_theta']).agg({
        'retrieval_cost_ms': ['mean', 'std']
    }).reset_index()
    
    for method in speed_summary['compression_method'].unique():
        method_data = speed_summary[speed_summary['compression_method'] == method]
        plt.errorbar(method_data['prompt_len_theta'], 
                    method_data['retrieval_cost_ms']['mean'],
                    yerr=method_data['retrieval_cost_ms']['std'],
                    label=method, marker='o', capsize=5)
    
    plt.xlabel('Prompt Length (θ)')
    plt.ylabel('Retrieval Latency (ms)')
    plt.title('Retrieval Speed by Compression Method')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'retrieval_speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Success Rate Heatmap
    plt.figure(figsize=(14, 10))
    
    success_pivot = df.pivot_table(
        index='model_name',
        columns='compression_method',
        values='is_semantically_equivalent',
        aggfunc='mean'
    )
    
    sns.heatmap(success_pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Semantic Success Rate'})
    plt.title('Semantic Success Rate by Model and Compression Method')
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare compression methods")
    parser.add_argument('--results-dirs', nargs='+', required=True,
                       help='Directories containing experiment results')
    parser.add_argument('--output-dir', type=str, default='results/combined_analysis',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Load all results
    print("Loading results from all experiments...")
    combined_df = load_and_label_results(args.results_dirs)
    
    # Save combined results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_dir / 'combined_results.csv', index=False)
    
    # Create visualizations
    print("Creating comparison visualizations...")
    create_comparison_plots(combined_df, args.output_dir)
    
    # Print summary statistics
    print("\n=== Compression Method Summary ===")
    summary = combined_df.groupby('compression_method').agg({
        'semantic_similarity': ['mean', 'std'],
        'retrieval_cost_ms': ['mean', 'std'],
        'is_semantically_equivalent': 'mean',
        'storage_cost_bytes': 'mean'
    }).round(4)
    print(summary)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()
