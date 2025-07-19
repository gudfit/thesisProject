import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset
import random
from copy import deepcopy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 128
learning_rate = 0.001

# Data transforms for MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Simple MLP model (another NN, as requested, for MNIST)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train function with collection budget (subsample fraction for separated task)
def train_model(collection_fraction=1.0):
    if collection_fraction < 1.0:
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        subset_indices = indices[:int(collection_fraction * len(train_dataset))]
        train_subset = Subset(train_dataset, subset_indices)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    return model

# Prune function (storage budget)
def prune_model(model, pruning_rate):
    pruned_model = deepcopy(model)
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
    return pruned_model

# Add Gaussian noise (retrieval budget)
def add_gaussian_noise(inputs, std):
    noise = torch.randn_like(inputs) * std
    return inputs + noise

# Evaluate with confidence threshold
def evaluate_model(model, noise_std, conf_threshold=0.7):
    model.eval()
    correct_high_conf = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = add_gaussian_noise(inputs, noise_std).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)
            high_conf_mask = max_probs > conf_threshold
            correct_high_conf += (predicted[high_conf_mask] == labels[high_conf_mask]).sum().item()
            total += labels.size(0)
    accuracy = correct_high_conf / total if total > 0 else 0
    return accuracy

# Experiment parameters
collection_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]  # Collection budgets (separated task: subsample data)
pruning_rates = [0.0, 0.2, 0.4, 0.6, 0.8]         # Storage
noise_stds = [0.0, 0.05, 0.1, 0.15, 0.2]          # Retrieval (scaled for MNIST)

# Run evaluations
results = np.zeros((len(collection_fractions), len(pruning_rates), len(noise_stds)))
models = {}

for c_idx, coll_frac in enumerate(collection_fractions):
    print(f'Training with collection fraction: {coll_frac}')
    model = train_model(coll_frac)
    models[coll_frac] = model
    for p_idx, prune_rate in enumerate(pruning_rates):
        pruned_model = prune_model(model, prune_rate)
        for n_idx, noise_std in enumerate(noise_stds):
            acc = evaluate_model(pruned_model, noise_std)
            results[c_idx, p_idx, n_idx] = acc
            print(f'Coll {coll_frac}, Prune {prune_rate}, Noise {noise_std}: Acc = {acc:.4f}')

# Simple analysis: Average accuracy per dimension to show patterns
print('Average acc per collection fraction:', np.mean(results, axis=(1,2)))
print('Average acc per pruning rate:', np.mean(results, axis=(0,2)))
print('Average acc per noise std:', np.mean(results, axis=(0,1)))

# "Proof" of separation: Compare integrated (full collection + high prune) vs separated (low collection + low prune)
# Assume "total budget" proxy: coll_frac + (1 - prune_rate) ~ constant, check if separated gives better acc per unit
integrated_acc = results[-1, -1, 0]  # Full collect, high prune, no noise
separated_acc = results[0, 0, 0]     # Low collect, no prune, no noise
print(f'Integrated acc (full collect + high prune): {integrated_acc}')
print(f'Separated acc (low collect + no prune): {separated_acc}')
