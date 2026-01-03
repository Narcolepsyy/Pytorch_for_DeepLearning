# PyTorch Workflow: Week 2 Cheat Sheet

## 1. Hardware Acceleration (Device Agnostic Code)
Always check for the best available hardware (GPU/MPS) to speed up training.
```python
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
```

## 2. Data Management
### Datasets & Transforms
Use `torchvision` for standard datasets and `transforms` for preprocessing.
- **`ToTensor()`**: Converts PIL images/NumPy arrays to tensors and scales pixels to $[0, 1]$.
- **`Normalize(mean, std)`**: Centers data around zero for stable gradients.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
```

### DataLoaders
Handles batching, shuffling, and multi-processing.
```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

## 3. Building Neural Networks
### Model Definition
Subclass `nn.Module` and define the `forward` pass.
- **`nn.Conv2d(in_channels, out_channels, kernel_size)`**: Extracts spatial features.
- **`nn.MaxPool2d(kernel_size)`**: Reduces spatial dimensions (downsampling).
- **`nn.Flatten()`**: Converts multi-dimensional tensors to a 1D vector before fully connected layers.

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 13 * 13, 10) # Example dimensions
        )

    def forward(self, x):
        return self.network(x)
```

## 4. Loss Functions & Optimizers
- **Classification**: Use `nn.CrossEntropyLoss()` (combines `LogSoftmax` and `NLLLoss`).
- **Optimizers**: 
  - `optim.SGD`: Stochastic Gradient Descent.
  - `optim.Adam`: Adaptive Moment Estimation (often faster convergence).

## 5. The Training & Evaluation Loop
### Training Loop
```python
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### Evaluation
Switch to `model.eval()` and use `torch.no_grad()` to disable gradient computation.
```python
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        # Calculate accuracy...
```

## 6. Key Metrics
- **Accuracy**: $\frac{\text{Correct Predictions}}{\text{Total Samples}}$
- **Loss**: Monitor both training and validation loss to detect **overfitting**.

