from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
import logging
import shutil
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ModelFineTuner:
    def __init__(self, model_id: str, output_dir: str, training_args: Dict[str, Any]):
        self.model_id = model_id
        self.output_dir = output_dir
        self.training_args = training_args

    def fine_tune(self, dataset_name: str, dataset_subset: str):
        logger.info(f"Fine-tuning model: {self.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset(dataset_name, dataset_subset)
        filtered_dataset = dataset.filter(
            lambda ex: len(ex["text"].strip()) > 0, num_proc=4
        )

        def tokenize_function(examples):
            examples["text"] = [text + tokenizer.eos_token for text in examples["text"]]
            return tokenizer(examples["text"], truncation=True, max_length=512)

        tokenized_dataset = filtered_dataset.map(
            tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
        )

        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model.resize_token_embeddings(len(tokenizer))
        temp_output_dir = f"./temp_training_{self.model_id.replace('/', '_')}"
        training_args = TrainingArguments(
            output_dir=temp_output_dir, overwrite_output_dir=True, **self.training_args
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        shutil.rmtree(temp_output_dir)
        logger.info(f"Fine-tuning complete. Model saved to {self.output_dir}")
