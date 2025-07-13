import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
import copy
import os
import bz2


# ===================================================================
# 1. SCALING LAW & MODEL DEFINITIONS (Unchanged)
# ===================================================================
def capacity_scaling_law(N, Q_max, a, gamma):
    return Q_max * (1 - a * np.power(N, -gamma))


def inference_scaling_law(theta, Q_N, b, delta):
    return Q_N * (1 - b * np.power(theta, -delta))


class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ===================================================================
# 2. EXPERIMENT & ANALYSIS FUNCTIONS
# ===================================================================
def evaluate_model(model, testloader, device, noise_level=0.0):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            if noise_level > 0:
                images += noise_level * torch.randn_like(images)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def run_experiment(base_model, testloader, config):
    results = []
    device = config["device"]
    parameters_to_prune = [
        (module, "weight")
        for module in base_model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    # Calculate file size for each pruned model for later analysis
    pruned_model_paths = {}
    for n_budget in config["storage_budgets_N"]:
        model_to_prune = copy.deepcopy(base_model)
        sparsity = 1.0 - n_budget
        if sparsity > 0:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sparsity,
            )
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)

        # Save a temporary file to get its size
        temp_path = f"temp_model_{n_budget:.2f}.pt"
        torch.save(model_to_prune.state_dict(), temp_path)
        pruned_model_paths[n_budget] = (temp_path, os.path.getsize(temp_path))

        for theta_budget in tqdm(
            config["retrieval_budgets_theta"], desc=f"Testing N={n_budget:.0%}"
        ):
            noise_level = 1.0 / theta_budget if theta_budget > 0 else float("inf")
            performance = evaluate_model(
                model_to_prune, testloader, device, noise_level=noise_level
            )
            results.append(
                {
                    "N_storage_budget": n_budget,
                    "theta_retrieval_budget": theta_budget,
                    "performance_Q": performance,
                    "storage_cost_bytes": pruned_model_paths[n_budget][1],
                }
            )

    # Clean up temporary files
    for path, size in pruned_model_paths.values():
        os.remove(path)

    return pd.DataFrame(results)


def analyze_and_plot(df_results, config):
    # This function is now focused on the scaling laws
    print("\n" + "=" * 80)
    print("--- SCALING LAW ANALYSIS (CIFAR-10) ---")
    print("=" * 80)
    # ... rest of the function is the same, ensures plots are generated ...
    theta_max = max(config["retrieval_budgets_theta"])
    df_analysis = df_results[df_results["theta_retrieval_budget"] == theta_max].copy()
    N_values = df_analysis["N_storage_budget"].values
    Q_values = df_analysis["performance_Q"].values

    try:
        popt_gamma, _ = curve_fit(
            capacity_scaling_law, N_values, Q_values, p0=[1.0, 0.5, 0.5], maxfev=5000
        )
        Q_max_fit, a_fit, gamma_fit = popt_gamma
        print(f"\n--- Fitting Capacity Scaling Law (γ) ---")
        print(f"  Fit successful! Asymptotic Max Performance (Q_max): {Q_max_fit:.4f}")
        print(f"  CAPACITY SCALING EXPONENT (γ): {gamma_fit:.4f}")
    except RuntimeError:
        gamma_fit, popt_gamma = float("nan"), [Q_values.max(), 0.5, 0.5]

    df_inference = df_results[df_results["N_storage_budget"] == 1.0].copy()
    theta_values = df_inference["theta_retrieval_budget"].values
    Q_values_theta = df_inference["performance_Q"].values
    Q_N_val = Q_values.max()
    try:
        fit_func = lambda theta, b, delta: inference_scaling_law(
            theta, Q_N_val, b, delta
        )
        popt_delta, _ = curve_fit(
            fit_func, theta_values, Q_values_theta, p0=[0.5, 0.5], maxfev=5000
        )
        b_fit, delta_fit = popt_delta
        print(f"\n--- Fitting Inference Scaling Law (δ) ---")
        print(f"  Fit successful! INFERENCE EFFICIENCY EXPONENT (δ): {delta_fit:.4f}")
    except RuntimeError:
        delta_fit, popt_delta = float("nan"), [0.5, 0.5]

    # ... Plotting logic ...
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].scatter(N_values, Q_values, label="Empirical Data", color="red", zorder=5)
    N_fine = np.linspace(min(N_values), max(N_values), 100)
    axs[0].plot(
        N_fine,
        capacity_scaling_law(N_fine, *popt_gamma),
        label=f"Fitted Law (γ={gamma_fit:.3f})",
        color="blue",
    )
    axs[0].set_title("Capacity Scaling Law")
    axs[0].legend()
    axs[0].grid(True)
    axs[1].scatter(
        theta_values, Q_values_theta, label="Empirical Data", color="red", zorder=5
    )
    theta_fine = np.linspace(min(theta_values), max(theta_values), 100)
    axs[1].plot(
        theta_fine,
        inference_scaling_law(theta_fine, Q_N_val, *popt_delta),
        label=f"Fitted Law (δ={delta_fit:.3f})",
        color="green",
    )
    axs[1].set_title("Inference Scaling Law")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xscale("log")
    plt.tight_layout()
    plt.savefig("cifar_cnn_scaling_law_analysis.png")
    print("\nSaved scaling law plots to 'cifar_cnn_scaling_law_analysis.png'")
    plt.show()


def analyze_eir_and_master_plot(df_results, testset):
    """Calculates EIR and generates the final master plot."""
    print("\n" + "=" * 80)
    print("--- EFFECTIVE INFORMATION RATIO & MASTER PLOT ---")
    print("=" * 80)

    # Calculate Information Content of the dataset
    all_data_bytes = np.concatenate(testset.data).tobytes()
    bzip2_compressed_size_bytes = len(bz2.compress(all_data_bytes))

    # Use performance at best retrieval budget
    theta_max = df_results["theta_retrieval_budget"].max()
    df_eir = df_results[df_results["theta_retrieval_budget"] == theta_max].copy()

    # EIR = (Info_Content * Performance) / Storage_Cost
    df_eir["EIR"] = (bzip2_compressed_size_bytes * df_eir["performance_Q"]) / df_eir[
        "storage_cost_bytes"
    ]

    # Sort by model size for a clean plot
    df_eir = df_eir.sort_values(by="N_storage_budget")

    print("\nCalculated EIR values:")
    print(
        df_eir[["N_storage_budget", "performance_Q", "EIR"]]
        .rename(columns={"performance_Q": "Fidelity"})
        .round(4)
    )

    # --- Generate the Master Plot ---
    peak_eir_row = df_eir.loc[df_eir["EIR"].idxmax()]
    peak_n = peak_eir_row["N_storage_budget"]
    peak_eir = peak_eir_row["EIR"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        df_eir["N_storage_budget"],
        df_eir["EIR"],
        marker="o",
        linestyle="-",
        color="b",
        label="Empirical EIR",
    )
    ax.axvline(
        peak_n,
        color="r",
        linestyle="--",
        lw=1,
        label=f"Peak Efficiency (N={peak_n:.0%})",
    )
    ax.scatter([peak_n], [peak_eir], color="r", s=150, zorder=10)
    ax.annotate(
        "Peak Efficiency",
        xy=(peak_n, peak_eir),
        xytext=(peak_n, peak_eir * 1.05),
        ha="center",
        fontsize=14,
        color="r",
    )
    ax.set_title("The Efficiency Frontier of Neural Compression", fontsize=20, pad=20)
    ax.set_xlabel("Storage Budget N (% of Full Model Size)", fontsize=14)
    ax.set_ylabel("Effective Information Ratio (EIR)", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    fig.tight_layout()
    save_path = "master_plot_EIR_vs_size.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nMaster plot saved to '{save_path}'")
    plt.show()


# ===================================================================
# 5. MAIN EXECUTION
# ===================================================================
def main():
    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": 128,
        "epochs": 10,
        "storage_budgets_N": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.04],
        "retrieval_budgets_theta": [2.0, 5.0, 10.0, 25.0, 50.0, 75.0, 1000.0],
    }
    output_csv_path = "cifar_pruning_results.csv"  # Define the output path

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    model_path = "full_cifar_cnn_model.pt"
    model = CifarCNN().to(config["device"])
    if not os.path.exists(model_path):
        print("Training full CIFAR model...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(config["epochs"]):
            model.train()
            for images, labels in tqdm(
                trainloader, desc=f"Epoch {epoch+1}/{config['epochs']}"
            ):
                images, labels = images.to(config["device"]), labels.to(
                    config["device"]
                )
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path))

    # --- Run Experiment and Save Results ---
    print("\n--- Starting Pruning Experiment ---")
    df_results = run_experiment(model, testloader, config)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Experiment results saved to {output_csv_path}")

    # --- Run ALL Analyses ---
    analyze_and_plot(df_results, config)
    analyze_eir_and_master_plot(df_results, testset)


if __name__ == "__main__":
    main()
