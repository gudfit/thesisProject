import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM
import copy
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ModelPruner:

    @staticmethod
    def get_prunable_parameters(model) -> List[Tuple[torch.nn.Module, str]]:
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, "weight"))
        if not parameters_to_prune:
            raise ValueError("No parameters found to prune in model")
        logger.info(f"Found {len(parameters_to_prune)} parameters to prune")
        return parameters_to_prune

    @staticmethod
    def prune_model(model, pruning_amount: float) -> torch.nn.Module:
        logger.info(f"Pruning model by {pruning_amount*100:.0f}%")
        pruned_model = copy.deepcopy(model)
        parameters_to_prune = ModelPruner.get_prunable_parameters(pruned_model)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_amount,
        )
        for module, param_name in parameters_to_prune:
            if prune.is_pruned(module):
                prune.remove(module, param_name)
        return pruned_model

    @staticmethod
    def create_pruned_models(
        base_model_path: str, output_dir: str, pruning_amounts: List[float]
    ) -> dict[str, str]:
        logger.info(f"Creating pruned models from {base_model_path}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        base_model.eval()
        pruned_models = {}
        original_path = Path(output_dir) / "pruned_0.pt"
        torch.save(base_model.state_dict(), original_path)
        pruned_models["0%"] = str(original_path)
        logger.info("Saved 0% pruned (original) model")
        for amount in pruning_amounts:
            pruned_model = ModelPruner.prune_model(base_model, amount)
            save_path = Path(output_dir) / f"pruned_{int(amount*100)}.pt"
            torch.save(pruned_model.state_dict(), save_path)
            pruned_models[f"{int(amount*100)}%"] = str(save_path)
            logger.info(f"Saved {amount*100:.0f}% pruned model")
            del pruned_model
        return pruned_models
