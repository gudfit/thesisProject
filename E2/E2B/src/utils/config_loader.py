# src/utils/config_loader.py
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ModelConfig:
    name: str
    model_id: str
    family: Optional[str] = None
    def __post_init__(self):
        if self.family is None:
            self.family = self._extract_family()
    def _extract_family(self) -> str:
        m = self.model_id.lower()
        f = {"gpt":"GPT","llama":"LLaMA","cerebras":"Cerebras","mistral":"Mistral","qwen":"Qwen","deepseek":"DeepSeek","phi":"Phi","gemma":"Gemma"}
        for k,v in f.items():
            if k in m:
                return v
        return "Other"

@dataclass
class ExperimentConfig:
    experiment_name: str
    dataset_name: str
    dataset_subset: Optional[str] = None
    test_split: str = "validation"
    ood_dataset_name: Optional[str] = None
    ood_dataset_subset: Optional[str] = None
    ood_split: str = "test"
    theta_budgets: List[int] = field(default_factory=lambda:[5,10,20,40,100])
    num_repetitions: int = 1
    semantic_threshold: float = 0.95
    semantic_threshold_ood: float = 0.85
    max_samples: int = 100
    max_samples_ood: Optional[int] = None
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
    pruning_amounts: List[float] = field(default_factory=lambda:[0.2,0.4,0.6,0.8])
    pruned_models_dir: str = "./pruned_models"
    finetuned_base_path: str = ""

@dataclass
class QuantizationExperimentConfig(ExperimentConfig):
    base_models: List[Dict[str,str]] = field(default_factory=list)
    quantization_configs: List[Dict[str,Any]] = field(default_factory=list)
    finetuned_models_dir: str = "./models/finetuned"
    quantized_models_dir: str = "./models/quantized"

@dataclass
class AdvancedPruningExperimentConfig(ExperimentConfig):
    base_models: List[Dict[str,str]] = field(default_factory=list)
    pruning_configs: List[Dict[str,Any]] = field(default_factory=list)
    finetuned_models_dir: str = "./models/finetuned"
    pruned_models_dir: str = "./models/advanced_pruned"

class ConfigLoader:
    @staticmethod
    def load_config(p: str) -> Dict[str, Any]:
        with open(p,"r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_finetune_config(p: str) -> FinetuneConfig:
        d = ConfigLoader.load_config(p)
        lbs = [ModelConfig(name=b["name"], model_id=b["model_id"]) for b in d.get("lambda_budgets",[])]
        return FinetuneConfig(
            experiment_name=d["experiment_name"],
            dataset_name=d["dataset_name"],
            dataset_subset=d.get("dataset_subset"),
            test_split=d.get("test_split","validation"),
            ood_dataset_name=d.get("ood_dataset_name"),
            ood_dataset_subset=d.get("ood_dataset_subset"),
            ood_split=d.get("ood_split","test"),
            theta_budgets=d.get("theta_budgets",[5,10,20,40,100]),
            num_repetitions=d.get("num_repetitions",1),
            semantic_threshold=d.get("semantic_threshold",0.95),
            semantic_threshold_ood=d.get("semantic_threshold_ood",0.85),
            max_samples=d.get("max_samples",100),
            max_samples_ood=d.get("max_samples_ood"),
            lambda_budgets=lbs,
            finetune_output_dir=d.get("finetune_output_dir","./finetuned_models"),
            training_args=d.get("training_args",{}),
            output_dir=d.get("output_dir","./results"),
        )

    @staticmethod
    def load_pruning_config(p: str) -> PruningConfig:
        d = ConfigLoader.load_config(p)
        return PruningConfig(
            experiment_name=d["experiment_name"],
            base_model_id=d["base_model_id"],
            dataset_name=d["dataset_name"],
            dataset_subset=d.get("dataset_subset"),
            test_split=d.get("test_split","validation"),
            ood_dataset_name=d.get("ood_dataset_name"),
            ood_dataset_subset=d.get("ood_dataset_subset"),
            ood_split=d.get("ood_split","test"),
            theta_budgets=d.get("theta_budgets",[5,10,20,40,100]),
            num_repetitions=d.get("num_repetitions",1),
            semantic_threshold=d.get("semantic_threshold",0.95),
            semantic_threshold_ood=d.get("semantic_threshold_ood",0.85),
            max_samples=d.get("max_samples",100),
            max_samples_ood=d.get("max_samples_ood"),
            pruning_amounts=d.get("pruning_amounts",[0.2,0.4,0.6,0.8]),
            pruned_models_dir=d.get("pruned_models_dir","./pruned_models"),
            finetuned_base_path=d.get("finetuned_base_path",""),
            output_dir=d.get("output_dir","./results"),
        )

    @staticmethod
    def load_quantization_config(p: str) -> QuantizationExperimentConfig:
        d = ConfigLoader.load_config(p)
        return QuantizationExperimentConfig(
            experiment_name=d["experiment_name"],
            dataset_name=d["dataset_name"],
            dataset_subset=d.get("dataset_subset"),
            test_split=d.get("test_split","validation"),
            ood_dataset_name=d.get("ood_dataset_name"),
            ood_dataset_subset=d.get("ood_dataset_subset"),
            ood_split=d.get("ood_split","test"),
            theta_budgets=d.get("theta_budgets",[5,10,20,40,100]),
            num_repetitions=d.get("num_repetitions",1),
            semantic_threshold=d.get("semantic_threshold",0.95),
            semantic_threshold_ood=d.get("semantic_threshold_ood",0.85),
            max_samples=d.get("max_samples",100),
            max_samples_ood=d.get("max_samples_ood"),
            base_models=d.get("base_models",[]),
            quantization_configs=d.get("quantization_configs",[]),
            finetuned_models_dir=d.get("finetuned_models_dir","./models/finetuned"),
            quantized_models_dir=d.get("quantized_models_dir","./models/quantized"),
            output_dir=d.get("output_dir","./results"),
        )

    @staticmethod
    def load_advanced_pruning_config(p: str) -> AdvancedPruningExperimentConfig:
        d = ConfigLoader.load_config(p)
        return AdvancedPruningExperimentConfig(
            experiment_name=d["experiment_name"],
            dataset_name=d["dataset_name"],
            dataset_subset=d.get("dataset_subset"),
            test_split=d.get("test_split","validation"),
            ood_dataset_name=d.get("ood_dataset_name"),
            ood_dataset_subset=d.get("ood_dataset_subset"),
            ood_split=d.get("ood_split","test"),
            theta_budgets=d.get("theta_budgets",[5,10,20,40,100]),
            num_repetitions=d.get("num_repetitions",1),
            semantic_threshold=d.get("semantic_threshold",0.95),
            semantic_threshold_ood=d.get("semantic_threshold_ood",0.85),
            max_samples=d.get("max_samples",100),
            max_samples_ood=d.get("max_samples_ood"),
            base_models=d.get("base_models",[]),
            pruning_configs=d.get("pruning_configs",[]),
            finetuned_models_dir=d.get("finetuned_models_dir","./models/finetuned"),
            pruned_models_dir=d.get("pruned_models_dir","./models/advanced_pruned"),
            output_dir=d.get("output_dir","./results"),
        )

