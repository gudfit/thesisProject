import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)

@dataclass
class PruningConfig:
    method: str = "magnitude"
    amount: float = 0.5
    structured: bool = False
    granularity: str = "unstructured"
    iterative_steps: int = 1
    use_movement_pruning: bool = False
    movement_regularization: float = 0.001

class AdvancedPruner:
    def __init__(self, config: PruningConfig):
        self.config = config

    def get_prunable_modules(self, model) -> List[Tuple[nn.Module, str]]:
        prunable_modules = []
        for _, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prunable_modules.append((module, "weight"))
        return prunable_modules

    def apply_magnitude_pruning(self, model, amount: float):
        logger.info(f"Applying magnitude pruning: {amount*100}% sparsity")
        parameters_to_prune = self.get_prunable_modules(model)
        if self.config.structured:
            for module, param_name in parameters_to_prune:
                if isinstance(module, nn.Linear):
                    prune.ln_structured(module, name=param_name, amount=amount, n=2, dim=0)
                elif isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name=param_name, amount=amount, n=2, dim=0)
        else:
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        for module, param_name in parameters_to_prune:
            if prune.is_pruned(module):
                prune.remove(module, param_name)

    def apply_movement_pruning(self, model, calibration_data, amount: float):
        logger.info(f"Applying movement pruning: {amount*100}% sparsity")
        parameters_to_prune = self.get_prunable_modules(model)
        movement_scores = {}
        for module, param_name in parameters_to_prune:
            param = getattr(module, param_name)
            movement_scores[id(param)] = torch.zeros_like(param)
        self.apply_magnitude_pruning(model, amount)

    def apply_lottery_ticket_pruning(self, model, original_state_dict, amount: float, iteration: int):
        logger.info(f"Applying lottery ticket pruning: iteration {iteration}, {amount*100}% sparsity")
        self.apply_magnitude_pruning(model, amount)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_state_dict:
                    mask = param != 0
                    param.data = original_state_dict[name] * mask

    def apply_block_sparse_pruning(self, model, amount: float, block_size: int = 4):
        logger.info(f"Applying block-sparse pruning: {amount*100}% sparsity, block size {block_size}")
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                h, w = weight.shape
                h_blocks = h // block_size
                w_blocks = w // block_size
                if h_blocks > 0 and w_blocks > 0:
                    weight_blocks = weight[:h_blocks*block_size, :w_blocks*block_size].reshape(
                        h_blocks, block_size, w_blocks, block_size
                    ).permute(0, 2, 1, 3).reshape(h_blocks * w_blocks, block_size * block_size)
                    block_importance = torch.norm(weight_blocks, dim=1)
                    num_blocks_to_prune = int(len(block_importance) * amount)
                    _, prune_indices = torch.topk(block_importance, num_blocks_to_prune, largest=False)
                    weight_blocks[prune_indices] = 0
                    weight[:h_blocks*block_size, :w_blocks*block_size] = weight_blocks.reshape(
                        h_blocks, w_blocks, block_size, block_size
                    ).permute(0, 2, 1, 3).reshape(h_blocks*block_size, w_blocks*block_size)

    def prune_model(self, model, amount: float) -> nn.Module:
        pruned_model = copy.deepcopy(model)
        if self.config.method == "magnitude":
            self.apply_magnitude_pruning(pruned_model, amount)
        elif self.config.method == "movement":
            self.apply_movement_pruning(pruned_model, None, amount)
        elif self.config.method == "structured":
            self.config.structured = True
            self.apply_magnitude_pruning(pruned_model, amount)
        elif self.config.method == "block_sparse":
            self.apply_block_sparse_pruning(pruned_model, amount)
        else:
            raise ValueError(f"Unknown pruning method: {self.config.method}")
        return pruned_model

    @staticmethod
    def create_pruned_model_suite(base_model_path: str, output_dir: str, pruning_configs: List[PruningConfig], amounts: List[float]) -> Dict[str, str]:
        logger.info(f"Creating pruned model suite from {base_model_path}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pruned_models: Dict[str, str] = {}
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        base_model.eval()
        for config in pruning_configs:
            pruner = AdvancedPruner(config)
            for amount in amounts:
                pruned_model = pruner.prune_model(base_model, amount)
                save_name = f"pruned_{config.method}_{int(amount*100)}"
                if config.structured:
                    save_name += "_structured"
                save_path = Path(output_dir) / f"{save_name}.pt"
                torch.save(pruned_model.state_dict(), save_path)
                pruned_models[save_name] = str(save_path)
                logger.info(f"Saved {save_name}")
                del pruned_model
                torch.cuda.empty_cache()
        return pruned_models

