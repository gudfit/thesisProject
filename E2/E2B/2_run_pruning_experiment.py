# E2/E2B/2_run_pruning_experiment.py
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


def reconstruct_sentence(model, tokenizer, sentence, prompt_len):
    device = model.device
    inputs = tokenizer(sentence, return_tensors="pt")
    full_ids = inputs.input_ids[0].to(device)
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
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = (time.time() - start_time) * 1000
    return tokenizer.decode(output_ids[0], skip_special_tokens=True), latency


def main():
    config = {
        "base_model_id": "gpt2-medium",
        "dataset_name": "wikitext",
        "dataset_subset": "wikitext-2-raw-v1",
        "test_split": "validation",
        "theta_budgets": [5, 10, 20, 40, 100],
        "semantic_threshold": 0.85,
        "num_repetitions": 1,
    }
    PRUNED_DIR = f"./pruned_models/{config['base_model_id'].replace('/', '_')}"
    RESULTS_PATH = "./results/gpt2_medium_pruning_results.csv"

    os.makedirs("./results", exist_ok=True)

    print("--- Running Pruning Experiment on GPT2-Medium ---")
    validation_sentences = get_sentences_from_dataset(config)
    validation_sentences = validation_sentences[:100]

    all_results = []
    pruned_files = sorted(
        os.listdir(PRUNED_DIR), key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    for file_name in pruned_files:
        model_path = os.path.join(PRUNED_DIR, file_name)
        model_name = f"GPT2-medium ({file_name.split('.')[0].replace('_', ' ')}d)"

        print(f"\nLoading model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(
            f"./finetuned_models/{config['base_model_id'].replace('/', '_')}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"./finetuned_models/{config['base_model_id'].replace('/', '_')}"
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        storage_cost = os.path.getsize(model_path)

        for sentence in tqdm(validation_sentences, desc=f"Testing {model_name}"):
            for theta in config["theta_budgets"]:
                latencies = []
                for _ in range(config["num_repetitions"]):
                    recon_text, latency = reconstruct_sentence(
                        model, tokenizer, sentence, theta
                    )
                    latencies.append(latency)

                avg_latency = np.mean(latencies)
                is_perfect = recon_text.strip() == sentence.strip()
                sem_sim = calculate_semantic_similarity(sentence, recon_text)

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
                    }
                )
        del model, tokenizer
        torch.cuda.empty_cache()

    pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)
    print(f"\nExperiment complete. Pruning results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
