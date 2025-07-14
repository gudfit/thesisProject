#!/usr/bin/env python3
"""
Main script to run experiments.

Usage:
    python scripts/run_experiment.py --config configs/experiments/finetune_multi_model.yaml --experiment finetune
    python scripts/run_experiment.py --config configs/experiments/pruning_gpt2.yaml --experiment pruning
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logging_utils import setup_logging
from src.experiments.finetune_experiment import FinetuneExperiment
from src.experiments.pruning_experiment import PruningExperiment

logger = logging.getLogger(__name__)

EXPERIMENT_CLASSES = {"finetune": FinetuneExperiment, "pruning": PruningExperiment}


def main():
    parser = argparse.ArgumentParser(description="Run LLM compression experiments")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["finetune", "pruning"],
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare models, don't run experiment",
    )
    parser.add_argument(
        "--skip-prepare", action="store_true", help="Skip model preparation"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger.info(f"Loading configuration from {args.config}")
    if args.experiment == "finetune":
        config = ConfigLoader.load_finetune_config(args.config)
    else:
        config = ConfigLoader.load_pruning_config(args.config)
    experiment_class = EXPERIMENT_CLASSES[args.experiment]
    experiment = experiment_class(config)
    try:
        if not args.skip_prepare:
            logger.info("Phase 1: Preparing models")
            experiment.prepare_models()

        if not args.prepare_only:
            logger.info("Phase 2: Running experiment")
            results = experiment.run_experiment()
            experiment.save_results(results)
            logger.info("Experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
