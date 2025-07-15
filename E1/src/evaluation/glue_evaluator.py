import torch
import numpy as np
import logging
from typing import Dict, List, Any, Union
from datasets import load_dataset, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import tempfile
import shutil

class GLUEEvaluator:
    TASK_CONFIGS = {
        "sst2": {"dataset": "glue", "subset": "sst2", "num_labels": 2, "metric": "accuracy", "text_fields": ["sentence"]},
        "mrpc": {"dataset": "glue", "subset": "mrpc", "num_labels": 2, "metric": "f1", "text_fields": ["sentence1", "sentence2"]},
        "rte": {"dataset": "glue", "subset": "rte", "num_labels": 2, "metric": "accuracy", "text_fields": ["sentence1", "sentence2"]},
        "cola": {"dataset": "glue", "subset": "cola", "num_labels": 2, "metric": "matthews_correlation", "text_fields": ["sentence"]},
        "qqp": {"dataset": "glue", "subset": "qqp", "num_labels": 2, "metric": "f1", "text_fields": ["question1", "question2"]},
    }

    def __init__(self, base_model_name: str, device: str = "cuda"):
        self.base_model_name = base_model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.logger = logging.getLogger(__name__)

    def evaluate_on_tasks(
        self,
        model: torch.nn.Module,
        tasks: List[str] = ["sst2", "mrpc", "rte"],
        train_samples: int = 1000,
        eval_samples: int = 500,
        epochs: int = 3,
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for task in tasks:
            self.logger.info(f"Evaluating on {task.upper()}...")
            try:
                score = self.evaluate_single_task(model, task, train_samples, eval_samples, epochs)
                results[task] = score
                self.logger.info(f"{task.upper()} score: {score:.4f}")
            except Exception as e:
                self.logger.error(f"Error evaluating {task}: {e}", exc_info=True)
                results[task] = 0.0
        results["average"] = float(np.mean(list(results.values()))) if results else 0.0
        return results

    def evaluate_single_task(
        self,
        model: torch.nn.Module,
        task: str,
        train_samples: int,
        eval_samples: int,
        epochs: int,
    ) -> float:
        if task not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task}")
        config = self.TASK_CONFIGS[task]
        
        from datasets import logging as dsets_logging
        dsets_logging.set_verbosity_error()

        dataset = load_dataset(config["dataset"], config["subset"])
        train_dataset = dataset["train"].shuffle(seed=42).select(range(min(train_samples, len(dataset["train"]))))
        eval_dataset = dataset["validation"].shuffle(seed=42).select(range(min(eval_samples, len(dataset["validation"]))))
        
        classifier = self._create_classifier_from_base(model, config["num_labels"])

        def preprocess_function(examples):
            args = (examples[field] for field in config["text_fields"])
            return self.tokenizer(*args, truncation=True, padding=True, max_length=128)

        tokenized_train = train_dataset.map(preprocess_function, batched=True)
        tokenized_eval = eval_dataset.map(preprocess_function, batched=True)
        
        if "label" in tokenized_train.column_names:
            tokenized_train = tokenized_train.rename_column("label", "labels")
            tokenized_eval = tokenized_eval.rename_column("label", "labels")
        if tokenized_train.features["labels"].dtype != "int64":
            features = tokenized_train.features.copy()
            features["labels"] = Value("int64")
            tokenized_train = tokenized_train.cast(features)
            tokenized_eval = tokenized_eval.cast(features)

        training_args = TrainingArguments(
            output_dir=f"./tmp/{task}_eval",
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
            fp16=torch.cuda.is_available(),
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            metric_name = config["metric"]
            if metric_name == "accuracy": return {"eval_accuracy": accuracy_score(labels, predictions)}
            if metric_name == "f1": return {"eval_f1": f1_score(labels, predictions, average="binary")}
            if metric_name == "matthews_correlation": return {"eval_matthews_correlation": matthews_corrcoef(labels, predictions)}
            return {}

        trainer = Trainer(
            model=classifier,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        eval_results = trainer.evaluate()
        
        main_metric = f"eval_{config['metric']}"
        score = eval_results.get(main_metric, 0.0)

        del classifier, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return float(score)

    def _create_classifier_from_base(self, base_model: torch.nn.Module, num_labels: int) -> torch.nn.Module:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_model.save_pretrained(temp_dir)
            classifier = AutoModelForSequenceClassification.from_pretrained(
                temp_dir,
                num_labels=num_labels,
            ).to(self.device)
            
        return classifier

    def evaluate_data_efficiency(
        self,
        model: torch.nn.Module,
        task: str = "sst2",
        sample_sizes: Union[List[int], None] = None,
    ) -> Dict[str, Any]:
        if sample_sizes is None:
            sample_sizes = [50, 100, 250, 500, 1000]
        results: Dict[str, Any] = {"sample_sizes": sample_sizes, "scores": [], "task": task}
        for n_samples in sample_sizes:
            self.logger.info(f"Evaluating with {n_samples} training samples...")
            score = self.evaluate_single_task(
                model, task, train_samples=n_samples, eval_samples=500, epochs=5
            )
            results["scores"].append(score)
        
        max_score = max(results["scores"]) if results["scores"] else 0
        threshold = 0.8 * max_score
        min_samples_needed = None
        for n_samples, score in zip(sample_sizes, results["scores"]):
            if score >= threshold:
                min_samples_needed = n_samples
                break
        
        results["min_samples_for_threshold"] = min_samples_needed
        results["threshold"] = threshold
        return results
