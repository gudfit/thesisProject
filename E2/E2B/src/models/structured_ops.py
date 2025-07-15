"""Structured pruning and quantisation utilities.

This module complements ``src/models/pruning.py`` with **channel-wise
structured pruning** (``torch.nn.utils.prune.ln_structured``) and a light‐
weight **dynamic INT8 quantisation** helper that relies on PyTorch’s built‑in
``torch.quantization.quantize_dynamic``.  The public surface mirrors the
existing :pyclass:`src.models.pruning.ModelPruner` API so you can drop these
functions into the current experiment pipeline with minimal changes.

Usage
-----
>>> from src.models.structured_ops import StructuredModelPruner, ModelQuantizer
>>> pruned_paths = StructuredModelPruner.create_pruned_models(
...     base_model_path="./models/finetuned/gpt2-medium",
...     output_dir="./models/pruned_structured/gpt2-medium",
...     pruning_amounts=[0.2, 0.4, 0.6, 0.8],
...     n=2, dim=0,
... )
>>> # Optional extra compression step
>>> quant_path = ModelQuantizer.quantize_and_save(
...     model_path=pruned_paths["80%"],  # the sparsest checkpoint
...     save_path="./models/quantised/gpt2-medium/int8_80pct",
... )

The output directory will contain weight files that you can load with
``AutoModelForCausalLM.from_pretrained`` exactly as before.  No changes are
required elsewhere in the code base other than switching the import to
``StructuredModelPruner`` if you want structured instead of unstructured
weights.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


###############################################################################
# Structured pruning ##########################################################
###############################################################################

class StructuredModelPruner:
    """Channel‑wise structured pruning helper.

    Parameters
    ----------
    n : int, optional
        The *n* norm to compute when ranking channels.  ``n=2`` (L2) tends to
        work well in practice.  Must be > 0.
    dim : int, optional
        The dimension along which entire channels will be removed.  For GPT‑2
        (and other decoder‑only transformers), ``dim=0`` corresponds to *output
        channels* and is what most structured‐pruning papers prune, but both
        axes are supported.
    """

    DEFAULT_NORM: int = 2
    DEFAULT_DIM: int = 0

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    @staticmethod
    def _apply_structured_pruning(
        module: torch.nn.Linear,
        amount: float,
        n: int = DEFAULT_NORM,
        dim: int = DEFAULT_DIM,
    ) -> None:
        """Apply in‑place Ln structured pruning to *one* ``nn.Linear`` layer."""
        prune.ln_structured(module, name="weight", amount=amount, n=n, dim=dim)
        prune.remove(module, "weight")  # make pruning permanent

    # ------------------------------------------------------------------
    # Public helpers that mirror ModelPruner
    # ------------------------------------------------------------------
    @staticmethod
    def prune_model(
        model: torch.nn.Module,
        pruning_amount: float,
        *,
        n: int = DEFAULT_NORM,
        dim: int = DEFAULT_DIM,
    ) -> torch.nn.Module:
        """Return a **deep‑copied** structured‑pruned version of *model*."""
        logger.info(
            "Structured‑pruning model by %d%% (n=%d, dim=%d)",
            int(pruning_amount * 100),
            n,
            dim,
        )
        pruned_model = copy.deepcopy(model)
        for module in pruned_model.modules():
            if isinstance(module, torch.nn.Linear):
                StructuredModelPruner._apply_structured_pruning(
                    module, pruning_amount, n=n, dim=dim
                )
        return pruned_model

    @staticmethod
    def create_pruned_models(
        *,
        base_model_path: str | Path,
        output_dir: str | Path,
        pruning_amounts: Iterable[float],
        n: int = DEFAULT_NORM,
        dim: int = DEFAULT_DIM,
        save_initial: bool = True,
    ) -> Dict[str, str]:
        """Materialise several structured‑pruned checkpoints to *output_dir*.

        Returns
        -------
        Dict[str, str]
            A mapping from a human‑readable key (e.g. ``"40%"``) to the saved
            *PyTorch* weight file path for ease of bookkeeping.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading base model from %s", base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        base_model.eval()

        pruned_models: Dict[str, str] = {}

        if save_initial:
            zero_path = output_dir / "pruned_0.pt"
            torch.save(base_model.state_dict(), zero_path)
            pruned_models["0%"] = str(zero_path)
            logger.info("Saved unpruned baseline to %s", zero_path)

        for amt in pruning_amounts:
            pct_label = f"{int(amt * 100)}%"
            logger.debug("Pruning %s of weights (structured)", pct_label)
            pruned = StructuredModelPruner.prune_model(
                base_model, amt, n=n, dim=dim
            )
            save_path = output_dir / f"pruned_{int(amt * 100)}.pt"
            torch.save(pruned.state_dict(), save_path)
            pruned_models[pct_label] = str(save_path)
            logger.info("Saved %s structured‑pruned model to %s", pct_label, save_path)
            del pruned
        return pruned_models


###############################################################################
# Dynamic INT8 quantisation ####################################################
###############################################################################

class ModelQuantizer:
    """Light‑weight dynamic INT8 quantisation utilities."""

    @staticmethod
    def quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
        """Return a *new* quantised copy of *model* using dynamic quantisation."""
        logger.info("Applying dynamic INT8 quantisation (linear layers only)…")
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
        )

    # ------------------------------------------------------------------
    # Convenience wrapper that operates on on‑disk checkpoints
    # ------------------------------------------------------------------
    @staticmethod
    def quantize_and_save(
        *,
        model_path: str | Path,
        save_path: str | Path,
        tokenizer_path: str | Path | None = None,
    ) -> str:
        """Load a model *from* ``model_path``, quantise, and **save** to *save_path*.

        The *tokenizer* is copied unchanged so downstream code can still call
        ``AutoTokenizer.from_pretrained`` on the same directory.
        """
        model_path = Path(model_path)
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info("Loading model for quantisation from %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()

        quantised = ModelQuantizer.quantize_dynamic(model)

        logger.info("Saving quantised model to %s", save_path)
        quantised.save_pretrained(save_path)

        # Copy / save tokenizer files if provided or discoverable
        tok_src = (
            Path(tokenizer_path) if tokenizer_path else model_path
        )  # fallback
        if (tok_src / "tokenizer.json").exists():
            tok = AutoTokenizer.from_pretrained(tok_src)
            tok.save_pretrained(save_path)

        return str(save_path)


__all__: List[str] = [
    "StructuredModelPruner",
    "ModelQuantizer",
]

