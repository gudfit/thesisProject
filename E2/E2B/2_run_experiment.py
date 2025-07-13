# E2/E2B/2_run_experiment.py
import yaml
import pandas as pd
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data_handler import get_sentences_from_dataset
from src.metrics import calculate_semantic_similarity


def reconstruct_sentence(model, tokenizer, sentence: str, prompt_len: int, device: str):
    """
    Reconstructs a sentence using the provided model.
    """
    target_device = model.device

    inputs = tokenizer(sentence, return_tensors="pt")
    full_ids = inputs.input_ids[0].to(target_device)

    if prompt_len >= len(full_ids):
        return sentence, 0.0

    prompt_ids = full_ids[:prompt_len].unsqueeze(0)
    attention_mask = torch.ones_like(prompt_ids)

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=len(full_ids) - prompt_len + 5,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = (time.time() - start_time) * 1000

    reconstructed_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return reconstructed_text, latency


def get_model_size_on_disk(model_path: str) -> int:
    total_size = 0
    for dirpath, _, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def main():
    with open("configs/finetune_experiment_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    results_dir = f"./results/{config['experiment_name']}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "raw_results.csv")

    print("--- Phase 2: Running Reconstruction Experiment ---")
    validation_sentences = get_sentences_from_dataset(config)
    validation_sentences = validation_sentences[:200]

    all_results = []
    for budget in config["lambda_budgets"]:
        model_name = budget["name"]
        model_id = budget["model_id"]
        model_name_safe = model_id.replace("/", "_")
        finetuned_path = os.path.join(config["finetune_output_dir"], model_name_safe)

        storage_cost = get_model_size_on_disk(finetuned_path)

        print(f"\nLoading fine-tuned model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
        model = AutoModelForCausalLM.from_pretrained(finetuned_path).to(device)

        for sentence in tqdm(validation_sentences, desc=f"Testing {model_name}"):
            for theta in config["theta_budgets"]:

                latencies = []
                for _ in range(config["num_repetitions"]):
                    recon_text, latency = reconstruct_sentence(
                        model, tokenizer, sentence, theta, device
                    )
                    latencies.append(latency)

                avg_latency = np.mean(latencies)

                is_perfect = recon_text.strip() == sentence.strip()
                sem_sim = calculate_semantic_similarity(sentence, recon_text)
                is_sem_eq = sem_sim >= config["semantic_threshold"]

                all_results.append(
                    {
                        "model_name": model_name,
                        "storage_cost_lambda": storage_cost,
                        "prompt_len_theta": theta,
                        "retrieval_cost_ms": avg_latency,
                        "original_sentence": sentence,
                        "reconstructed_sentence": recon_text,
                        "is_perfect": is_perfect,
                        "semantic_similarity": sem_sim,
                        "is_semantically_equivalent": is_sem_eq,
                    }
                )
        del model, tokenizer
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)
    print(f"\nExperiment complete. Raw results saved to {results_path}")


if __name__ == "__main__":
    main()
