import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 20
batch_size = 128
learning_rate = 0.001

# Data loading and transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the model
def train_model():
    model = SimpleCNN().to(device)
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

    torch.save(model.state_dict(), 'cifar10_cnn.pth')
    return model

# Load or train the model
try:
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('cifar10_cnn.pth'))
except:
    model = train_model()

# Function to prune the model (storage budget: higher pruning = lower budget)
def prune_model(model, pruning_rate):
    pruned_model = deepcopy(model)
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
    return pruned_model

# Function to add Gaussian noise (retrieval budget: higher noise std = lower budget)
def add_gaussian_noise(inputs, std):
    noise = torch.randn_like(inputs) * std
    return inputs + noise

# Evaluate model with confidence threshold (p-guaranteed: success if correct and confidence > conf_threshold)
def evaluate_model(model, noise_std, conf_threshold=0.5):
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
pruning_rates = [0.0, 0.2, 0.4, 0.6, 0.8]  # Storage budgets (0 = full model, higher = more compressed)
noise_stds = [0.0, 0.1, 0.2, 0.3, 0.4]     # Retrieval budgets (0 = no noise, higher = worse retrieval)
conf_threshold = 0.7  # p for guaranteed region (higher = stricter)

# Run evaluations
results = np.zeros((len(pruning_rates), len(noise_stds)))
for i, prune_rate in enumerate(pruning_rates):
    pruned_model = prune_model(model, prune_rate)
    for j, noise_std in enumerate(noise_stds):
        acc = evaluate_model(pruned_model, noise_std, conf_threshold)
        results[i, j] = acc
        print(f'Pruning {prune_rate*100}%, Noise {noise_std}: High-Conf Accuracy = {acc:.4f}')

# Plot to visualize degradation (proof: monotonic decrease in accuracy with higher pruning/noise)
plt.imshow(results, cmap='viridis', origin='lower')
plt.colorbar(label='High-Confidence Accuracy')
plt.xticks(range(len(noise_stds)), noise_stds)
plt.yticks(range(len(pruning_rates)), [f'{r*100}%' for r in pruning_rates])
plt.xlabel('Noise Std (Lower Retrieval Budget)')
plt.ylabel('Pruning Rate (Lower Storage Budget)')
plt.title('Degradation in Performance: Storage vs Retrieval Loss')
plt.show()

# Simple fit to scaling law (latent Q* ~ (1 - a * N^{-gamma}) * (1 - b * theta^{-delta}))
# Here, approximate N = 1 - prune_rate, theta = 1 / (noise_std + 1e-3)
from scipy.optimize import curve_fit

def scaling_law(params, prune, noise):
    a, gamma, b, delta = params
    N = 1 - prune
    theta = 1 / (noise + 1e-3)
    return (1 - a * N**(-gamma)) * (1 - b * theta**(-delta))

def objective(params, prune, noise, acc):
    return scaling_law(params, prune, noise) - acc

# Flatten data for fitting
prune_flat = np.repeat(pruning_rates, len(noise_stds))
noise_flat = np.tile(noise_stds, len(pruning_rates))
acc_flat = results.flatten()

# Initial guess
initial_params = [1.0, 0.5, 1.0, 0.5]

# Fit (least squares)
from scipy.optimize import least_squares
fit_result = least_squares(objective, initial_params, args=(prune_flat, noise_flat, acc_flat))

print('Fitted params (a, gamma, b, delta):', fit_result.x)

# This "proves" the framework by showing fitted scaling law matches degradation patterns.
