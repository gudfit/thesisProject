#!/usr/bin/env python3
"""
Run basic pruning experiment on GPT-2 Medium (matching your example output).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.experiments.pruning_experiment import PruningExperiment
from src.utils.config_loader import PruningConfig

config = PruningConfig(
    experiment_name="gpt2_medium_pruning_experiment",
    base_model_id="gpt2-medium",
    dataset_name="wikitext",
    dataset_subset="wikitext-2-raw-v1",
    test_split="validation",
    theta_budgets=[5, 10, 20, 40, 100],
    num_repetitions=1,
    semantic_threshold=0.95,
    pruning_amounts=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95],
    pruned_models_dir="./models/pruned/gpt2-medium",
    finetuned_base_path="./models/finetuned/gpt2-medium",
    max_samples= 100,
    output_dir="./results"
)

experiment = PruningExperiment(config)
print("Running GPT-2 Medium pruning experiment...")
experiment.prepare_models()
results = experiment.run_experiment()
experiment.save_results(results)
print("Done!")
