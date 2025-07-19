# toy.py

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

# Create data directory if it doesn't exist
if not os.path.exists('./data'):
    os.makedirs('./data')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loaders for CIFAR-10
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train function
def train_model(model, epochs=20, from_scratch=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001 if not from_scratch else 0.001)  # Same LR for simplicity
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return model

# Lottery ticket pruning: Prune based on magnitude, reset to initial weights (masked), retrain from scratch
def lottery_prune_model(model, sparsity, init_model):
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weight = module.weight.data
            thresh = np.percentile(torch.abs(weight).cpu().numpy(), sparsity * 100)
            mask = (torch.abs(weight) >= thresh).float()
            masks[name] = mask
    # Reset to initial weights, apply mask
    model.load_state_dict(init_model.state_dict())
    for name, module in model.named_modules():
        if name in masks:
            module.weight.data *= masks[name]
    return model

# Fine-tune after pruning (for lottery: retrain from scratch)
def fine_tune_model(model, epochs=20):  # Longer for lottery retrain
    return train_model(model, epochs=epochs, from_scratch=True)

# Evaluation with noise (also returns confidences for expected fidelity)
def evaluate_model(model, testloader, device, noise_level=0.0):
    model.eval()
    correct = 0
    total = 0
    confidences = []  # For expected fidelity (mean max softmax prob for correct preds)
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if noise_level > 0:
                inputs += noise_level * torch.randn_like(inputs)
                inputs = torch.clamp(inputs, -1, 1)  # Clamp to [-1,1] after noise
            outputs = model(inputs)
            probs = softmax(outputs)
            max_probs, predicted = torch.max(probs, 1)
            total += labels.size(0)
            correct_mask = (predicted == labels)
            correct += correct_mask.sum().item()
            confidences.extend(max_probs[correct_mask].cpu().numpy())  # Conf for correct only
    accuracy = 100 * correct / total if total > 0 else 0
    expected_fidelity = np.mean(confidences) if confidences else 0  # Mean conf for successes
    return accuracy, expected_fidelity

# Compute storage cost as file size of saved state_dict
def get_storage_cost(model):
    temp_path = "temp_model.pt"
    torch.save(model.state_dict(), temp_path)
    size = os.path.getsize(temp_path)
    os.remove(temp_path)
    return size

# Main experiment
num_trials = 10
pruning_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Sparsity from 0% to 90%
noise_levels = [0, 0.05, 0.1, 0.2, 0.3, 0.5]  # Sigma values

results_acc = {prune: {noise: [] for noise in noise_levels} for prune in pruning_levels}
results_fid = {prune: {noise: [] for noise in noise_levels} for prune in pruning_levels}
storage_costs = {prune: [] for prune in pruning_levels}  # Track storage cost per prune level

for trial in tqdm(range(num_trials), desc="Trials"):
    model = SimpleCNN().to(device)
    init_model = copy.deepcopy(model)  # Save initial weights for lottery reset
    model = train_model(model)
    
    for prune in pruning_levels:
        pruned_model = copy.deepcopy(model)
        if prune > 0:
            pruned_model = lottery_prune_model(pruned_model, prune, init_model)
            pruned_model = fine_tune_model(pruned_model)
        
        storage_costs[prune].append(get_storage_cost(pruned_model))
        
        for noise in noise_levels:
            acc, fid = evaluate_model(pruned_model, testloader, device, noise)
            results_acc[prune][noise].append(acc)
            results_fid[prune][noise].append(fid)

# Compute means and stds
mean_acc = {prune: {noise: np.mean(results_acc[prune][noise]) for noise in noise_levels} for prune in pruning_levels}
std_acc = {prune: {noise: np.std(results_acc[prune][noise]) for noise in noise_levels} for prune in pruning_levels}
mean_fid = {prune: {noise: np.mean(results_fid[prune][noise]) for noise in noise_levels} for prune in pruning_levels}
std_fid = {prune: {noise: np.std(results_fid[prune][noise]) for noise in noise_levels} for prune in pruning_levels}
mean_storage = {prune: np.mean(storage_costs[prune]) for prune in pruning_levels}

# Compute I_ref automatically with bz2 compression
all_data_bytes = np.concatenate([img for img, _ in trainset]).tobytes()
I_ref = len(bz2.compress(all_data_bytes))
print(f"I_ref (compressed bits in CIFAR-10 train): {I_ref * 8}")  # Convert bytes to bits

# Compute and print metrics
lambda_max_acc = mean_acc[0][0] / 100  # Q^* at max (unpruned, no noise)
print("\nKey Results:")
for prune in pruning_levels:
    for noise in noise_levels:
        success_rate = mean_acc[prune][noise] / 100  # As fraction
        degradation_rate = 1 - (success_rate / lambda_max_acc) if lambda_max_acc > 0 else 0
        expected_fidelity = mean_fid[prune][noise]
        storage_cost = mean_storage[prune]
        eir = (I_ref * success_rate) / storage_cost if storage_cost > 0 else 0
        
        print(f"\nFor prune={prune*100:.0f}%, noise sigma={noise}:")
        print(f"Success Rate (Accuracy): {success_rate*100:.1f}% ± {std_acc[prune][noise]:.1f}%")
        print(f"Degradation Rate: {degradation_rate:.3f}")
        print(f"Expected Fidelity (Mean Conf for Correct): {expected_fidelity:.3f} ± {std_fid[prune][noise]:.3f}")
        print(f"EIR: {eir:.2e}")

# Automatic fluctuation check for non-monotonicity in acc vs prune (no noise)
print("\nFluctuation Check (no noise):")
prune_sorted = sorted(pruning_levels)
acc_no_noise = [mean_acc[p][0] for p in prune_sorted]
upticks = []
for i in range(1, len(prune_sorted)):
    if acc_no_noise[i] > acc_no_noise[i-1]:
        uptick = acc_no_noise[i] - acc_no_noise[i-1]
        upticks.append((prune_sorted[i-1], prune_sorted[i], uptick))
        print(f"Uptick from {prune_sorted[i-1]*100:.0f}% to {prune_sorted[i]*100:.0f}% pruning: {uptick:.1f}%")
if not upticks:
    print("No upticks observed; accuracy monotonically non-increasing with pruning.")

# Max normalised delta
all_accs = [mean_acc[p][n] for p in pruning_levels for n in noise_levels]
max_acc = max(all_accs)
min_acc = min(all_accs)
max_delta = (max_acc - min_acc) / max_acc if max_acc > 0 else 0
print(f"Max normalised delta: {max_delta:.2f}")

# Save full results to CSV
df_results = pd.DataFrame([
    {
        "prune_sparsity": prune,
        "noise_sigma": noise,
        "mean_acc": mean_acc[prune][noise],
        "std_acc": std_acc[prune][noise],
        "mean_fid": mean_fid[prune][noise],
        "std_fid": std_fid[prune][noise],
        "mean_storage_bytes": mean_storage[prune]
    }
    for prune in pruning_levels for noise in noise_levels
])
df_results.to_csv('cifar_results.csv', index=False)
print("Results saved to cifar_results.csv")
