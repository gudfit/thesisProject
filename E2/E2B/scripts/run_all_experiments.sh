#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=python

$PY scripts/run_experiment.py --config configs/experiments/advanced_pruning_experiment.yaml --experiment advanced_pruning --log-level INFO

$PY scripts/run_experiment.py --config configs/experiments/quantization_experiment.yaml --experiment quantization --log-level INFO

$PY scripts/compare_compression_methods.py \
  --results-dirs \
    results/multi_model_finetune_comparison \
    results/gpt2_medium_pruning_analysis \
    results/advanced_pruning_comparison \
    results/sparsegpt_quantization_analysis \
  --output-dir results/comprehensive_analysis \
  --semantic-threshold 0.95

