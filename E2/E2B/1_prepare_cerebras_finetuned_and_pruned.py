# 1_prepare_gpt2_finetuned_and_pruned.py
import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import copy
import yaml

# --- Fine-Tuning Function ---
def fine_tune_model(model_id, finetuned_path):
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from datasets import load_dataset
    print(f"\n--- Fine-tuning model: {model_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    filtered_dataset = dataset.filter(lambda ex: len(ex['text'].strip()) > 0, num_proc=4)
    
    def tokenize(ex):
        ex['text'] = [t + tokenizer.eos_token for t in ex['text']]
        return tokenizer(ex["text"], truncation=True, max_length=512)
        
    tokenized_dataset = filtered_dataset.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(tokenizer))
    
    training_args_dict = {
        "per_device_train_batch_size": 4,
        "num_train_epochs": 1,
        "logging_steps": 200,
        "overwrite_output_dir": True,
        "output_dir": f"./temp_training_{model_id.replace('/', '_')}"
    }
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_dict),
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    trainer.save_model(finetuned_path)
    tokenizer.save_pretrained(finetuned_path)
    
    import shutil
    shutil.rmtree(training_args_dict["output_dir"])

def main():
    BASE_MODEL_ID = "cerebras/Cerebras-GPT-111M"
    FINETUNED_DIR = "./finetuned_models"
    PRUNED_DIR = f"./pruned_models/{BASE_MODEL_ID.replace('/', '_')}"
    PRUNING_AMOUNTS = [0.2, 0.4, 0.6, 0.8]

    os.makedirs(FINETUNED_DIR, exist_ok=True)
    os.makedirs(PRUNED_DIR, exist_ok=True)

    finetuned_path = os.path.join(FINETUNED_DIR, BASE_MODEL_ID.replace('/', '_'))

    if not os.path.exists(finetuned_path):
        fine_tune_model(BASE_MODEL_ID, finetuned_path)
    else:
        print(f"Found existing fine-tuned model at {finetuned_path}")

    print(f"\nLoading fine-tuned base model from {finetuned_path}")
    base_model = AutoModelForCausalLM.from_pretrained(finetuned_path)
    base_model.eval()

    # --- THIS IS THE FIX ---
    parameters_to_prune = []
    for name, module in base_model.named_modules():
        # This is a more robust way to find all linear layers with weights
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    if not parameters_to_prune:
        print("ERROR: No parameters found to prune. Check model architecture and layer types.")
        return
    # --- END OF FIX ---

    torch.save(base_model.state_dict(), os.path.join(PRUNED_DIR, "pruned_0.pt"))
    print("Saved 0% pruned (original fine-tuned) model.")

    for amount in PRUNING_AMOUNTS:
        print(f"Creating model with {amount*100:.0f}% pruning...")
        model_to_prune = copy.deepcopy(base_model)
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        
        for module, param_name in parameters_to_prune:
            if prune.is_pruned(module):
                prune.remove(module, param_name)
        
        pruned_save_path = os.path.join(PRUNED_DIR, f"pruned_{int(amount*100)}.pt")
        torch.save(model_to_prune.state_dict(), pruned_save_path)
        print(f"Saved {amount*100:.0f}% pruned model.")

    print("\n--- Pruned model preparation complete. ---")

if __name__ == "__main__":
    main()
