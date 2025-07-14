#!/usr/bin/env python3
"""
Analyze experiment results and create visualizations.

Usage:
    python scripts/analyze_results.py --results-path results/experiment_name
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.analyzers import FinetuneAnalyzer, PruningAnalyzer
from src.analysis.visualization import ExperimentVisualizer
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--results-path", type=str, required=True, help="Path to results directory"
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        default="auto",
        choices=["auto", "finetune", "pruning"],
        help="Type of analysis to perform",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="all",
        choices=["all", "plots", "tables"],
        help="Output format",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    results_path = Path(args.results_path)
    if not results_path.exists():
        logger.error(f"Results path not found: {results_path}")
        sys.exit(1)

    results_file = results_path / "raw_results.csv"
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        sys.exit(1)

    logger.info(f"Loading results from {results_file}")
    import pandas as pd

    results_df = pd.read_csv(results_file)
    if args.analysis_type == "auto":
        if "pruned" in str(results_path).lower():
            args.analysis_type = "pruning"
        else:
            args.analysis_type = "finetune"

    logger.info(f"Performing {args.analysis_type} analysis")

    class DummyConfig:
        semantic_threshold = 0.95
        theta_budgets = [5, 10, 20, 40, 100]

    config = DummyConfig()

    if args.analysis_type == "finetune":
        analyzer = FinetuneAnalyzer(results_df, config)
    else:
        analyzer = PruningAnalyzer(results_df, config)

    if args.output_format in ["all", "tables"]:
        logger.info("Computing summary statistics...")
        summary = analyzer.compute_summary_statistics()
        print("\n=== Summary Statistics ===")
        print(summary)

        logger.info("Computing retrieval degradation...")
        degradation = analyzer.compute_retrieval_degradation()
        print("\n=== Retrieval Degradation ===")
        print(degradation)

        if args.analysis_type == "finetune":
            efficiency = analyzer.compute_model_efficiency()
            print("\n=== Model Efficiency ===")
            print(efficiency)
        else:
            pruning_impact = analyzer.compute_pruning_impact()
            print("\n=== Pruning Impact ===")
            print(pruning_impact)

    if args.output_format in ["all", "plots"]:
        logger.info("Creating visualizations...")
        visualizer = ExperimentVisualizer()
        summary = analyzer.compute_summary_statistics()
        visualizer.plot_performance_comparison(
            summary, results_path / "performance_comparison.png"
        )

        degradation = analyzer.compute_retrieval_degradation()
        visualizer.plot_retrieval_degradation(
            degradation, results_path / "retrieval_degradation.png"
        )

        if args.analysis_type == "finetune":
            efficiency = analyzer.compute_model_efficiency()
            visualizer.plot_efficiency_scatter(
                efficiency, results_path / "efficiency_scatter.png"
            )

        logger.info(f"Visualizations saved to {results_path}")

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
