import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    name: str
    model_id: str
    family: Optional[str] = None

    def __post_init__(self):
        if self.family is None:
            self.family = self._extract_family()

    def _extract_family(self) -> str:
        model_lower = self.model_id.lower()
        families = {
            "gpt": "GPT",
            "llama": "LLaMA",
            "cerebras": "Cerebras",
            "mistral": "Mistral",
            "qwen": "Qwen",
            "deepseek": "DeepSeek",
            "phi": "Phi",
            "gemma": "Gemma",
        }
        for key, family in families.items():
            if key in model_lower:
                return family
        return "Other"


@dataclass
class ExperimentConfig:
    experiment_name: str
    dataset_name: str
    dataset_subset: str
    test_split: str = "validation"
    theta_budgets: List[int] = field(default_factory=lambda: [5, 10, 20, 40, 100])
    num_repetitions: int = 1
    semantic_threshold: float = 0.95
    output_dir: str = "./results"

    def get_results_path(self) -> Path:
        return Path(self.output_dir) / self.experiment_name


@dataclass
class FinetuneConfig(ExperimentConfig):
    lambda_budgets: List[ModelConfig] = field(default_factory=list)
    finetune_output_dir: str = "./finetuned_models"
    training_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PruningConfig(ExperimentConfig):
    base_model_id: str = ""
    pruning_amounts: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    pruned_models_dir: str = "./pruned_models"
    finetuned_base_path: str = ""

@dataclass
class QuantizationExperimentConfig(ExperimentConfig):
    """Configuration for quantization experiments."""
    base_models: List[Dict[str, str]] = field(default_factory=list)
    quantization_configs: List[Dict[str, Any]] = field(default_factory=list)
    finetuned_models_dir: str = "./models/finetuned"
    quantized_models_dir: str = "./models/quantized"
    max_samples: int = 1000

@dataclass
class AdvancedPruningExperimentConfig(ExperimentConfig):
    """Configuration for advanced pruning experiments."""
    base_models: List[Dict[str, str]] = field(default_factory=list)
    pruning_configs: List[Dict[str, Any]] = field(default_factory=list)
    finetuned_models_dir: str = "./models/finetuned"
    pruned_models_dir: str = "./models/advanced_pruned"
    max_samples: int = 1000

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_finetune_config(config_path: str) -> FinetuneConfig:
        data = ConfigLoader.load_config(config_path)
        lambda_budgets = [
            ModelConfig(name=budget["name"], model_id=budget["model_id"])
            for budget in data.get("lambda_budgets", [])
        ]

        return FinetuneConfig(
            experiment_name=data["experiment_name"],
            dataset_name=data["dataset_name"],
            dataset_subset=data["dataset_subset"],
            test_split=data.get("test_split", "validation"),
            theta_budgets=data.get("theta_budgets", [5, 10, 20, 40, 100]),
            num_repetitions=data.get("num_repetitions", 1),
            semantic_threshold=data.get("semantic_threshold", 0.95),
            lambda_budgets=lambda_budgets,
            finetune_output_dir=data.get("finetune_output_dir", "./finetuned_models"),
            training_args=data.get("training_args", {}),
        )

    @staticmethod
    def load_pruning_config(config_path: str) -> PruningConfig:
        data = ConfigLoader.load_config(config_path)
        return PruningConfig(
            experiment_name=data["experiment_name"],
            base_model_id=data["base_model_id"],
            dataset_name=data["dataset_name"],
            dataset_subset=data["dataset_subset"],
            test_split=data.get("test_split", "validation"),
            theta_budgets=data.get("theta_budgets", [5, 10, 20, 40, 100]),
            num_repetitions=data.get("num_repetitions", 1),
            semantic_threshold=data.get("semantic_threshold", 0.95),
            pruning_amounts=data.get("pruning_amounts", [0.2, 0.4, 0.6, 0.8]),
            pruned_models_dir=data.get("pruned_models_dir", "./pruned_models"),
            finetuned_base_path=data.get("finetuned_base_path", ""),
        )

    @staticmethod
    def load_quantization_config(config_path: str) -> QuantizationExperimentConfig:
        """Load quantization experiment configuration."""
        data = ConfigLoader.load_config(config_path)
        
        return QuantizationExperimentConfig(
            experiment_name=data['experiment_name'],
            dataset_name=data['dataset_name'],
            dataset_subset=data['dataset_subset'],
            test_split=data.get('test_split', 'validation'),
            theta_budgets=data.get('theta_budgets', [5, 10, 20, 40, 100]),
            num_repetitions=data.get('num_repetitions', 1),
            semantic_threshold=data.get('semantic_threshold', 0.95),
            base_models=data.get('base_models', []),
            quantization_configs=data.get('quantization_configs', []),
            finetuned_models_dir=data.get('finetuned_models_dir', './models/finetuned'),
            quantized_models_dir=data.get('quantized_models_dir', './models/quantized'),
            max_samples=data.get('max_samples', 1000)
        )
    
    @staticmethod
    def load_advanced_pruning_config(config_path: str) -> AdvancedPruningExperimentConfig:
        """Load advanced pruning experiment configuration."""
        data = ConfigLoader.load_config(config_path)
        
        return AdvancedPruningExperimentConfig(
            experiment_name=data['experiment_name'],
            dataset_name=data['dataset_name'],
            dataset_subset=data['dataset_subset'],
            test_split=data.get('test_split', 'validation'),
            theta_budgets=data.get('theta_budgets', [5, 10, 20, 40, 100]),
            num_repetitions=data.get('num_repetitions', 1),
            semantic_threshold=data.get('semantic_threshold', 0.95),
            base_models=data.get('base_models', []),
            pruning_configs=data.get('pruning_configs', []),
            finetuned_models_dir=data.get('finetuned_models_dir', './models/finetuned'),
            pruned_models_dir=data.get('pruned_models_dir', './models/advanced_pruned'),
            max_samples=data.get('max_samples', 1000)
        )
