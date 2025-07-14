# 1_prepare_models.py
import yaml
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def fine_tune_model(model_id, finetuned_path, dataset, training_args_dict):
    print(f"\n--- Fine-tuning model: {model_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # No need to save here, the final save_pretrained will handle it.
    
    # --- THIS IS THE FIX ---
    # Filter out empty or whitespace-only lines BEFORE tokenization.
    # The Trainer needs non-empty examples to function.
    print("Filtering empty lines from the dataset...")
    filtered_dataset = dataset.filter(lambda example: len(example['text'].strip()) > 0)
    print("Filtering complete.")

    def tokenize_function(examples):
        # We also add a newline to the end of each text to help the model learn sentence boundaries.
        examples['text'] = [text + tokenizer.eos_token for text in examples['text']]
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_dataset = filtered_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    # --- END OF FIX ---
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    # It's good practice to resize embeddings if the pad token was added
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=f"./temp_training_{model_id.replace('/', '_')}",
        overwrite_output_dir=True,
        **training_args_dict
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )
    
    trainer.train()
    print(f"Fine-tuning complete. Saving model to {finetuned_path}")
    trainer.save_model(finetuned_path)
    tokenizer.save_pretrained(finetuned_path)
    
    import shutil
    shutil.rmtree(f"./temp_training_{model_id.replace('/', '_')}")
def main():
    with open('configs/finetune_experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config['finetune_output_dir'], exist_ok=True)
    
    print("--- Phase 1: Preparing All Models (Fine-tuning) ---")
    dataset = load_dataset(config['dataset_name'], config['dataset_subset'])
    
    for budget in config['lambda_budgets']:
        model_id = budget['model_id']
        model_name_safe = model_id.replace("/", "_")
        finetuned_path = os.path.join(config['finetune_output_dir'], model_name_safe)
        
        if not os.path.exists(finetuned_path):
            fine_tune_model(model_id, finetuned_path, dataset, config['training_args'])
        else:
            print(f"Found existing fine-tuned model for {model_id} at {finetuned_path}")

    print("\n--- Model preparation complete. ---")

if __name__ == "__main__":
    main()
