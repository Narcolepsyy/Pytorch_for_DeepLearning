# PyTorch Fundamentals: Week 1 Cheat Sheet

## 1. Introduction to PyTorch
PyTorch is a flexible deep learning framework known for its **dynamic computation graphs** (Eager Execution).
- **Dynamic Graphs**: The graph is built on-the-fly as operations are executed, allowing for easier debugging and conditional logic (if/else) within the model.
- **GPU Acceleration**: Seamlessly move computations from CPU to GPU using `.to('cuda')`.

## 2. Tensors: The Building Blocks
Tensors are multi-dimensional arrays, similar to NumPy's `ndarray`, but optimized for deep learning.

### Creation
```python
import torch
import numpy as np

# From a list
x = torch.tensor([1, 2, 3])

# From NumPy
arr = np.array([1, 2, 3])
x = torch.from_numpy(arr)

# Predefined values
zeros = torch.zeros(2, 3)  # 2x3 tensor of 0s
ones = torch.ones(2, 3)    # 2x3 tensor of 1s
rand = torch.rand(2, 3)    # 2x3 tensor of random values [0, 1)
```

### Key Attributes & Operations
- **Attributes**: `x.shape` (size), `x.dtype` (data type), `x.device` (CPU/GPU).
- **Reshaping**: `x.view(rows, cols)` or `x.reshape(rows, cols)`. Use `-1` to let PyTorch infer a dimension.
- **Math**: `x + y`, `torch.matmul(x, y)` (matrix multiplication), `x.mean()`, `x.std()`.

## 3. Data Preprocessing
### Normalization (Standardization)
Standardizing data helps the model converge faster by keeping gradients stable.
$$z = \frac{x - \mu}{\sigma}$$
```python
mean = x.mean()
std = x.std()
x_norm = (x - mean) / std
```

## 4. Building Neural Networks
Models are defined by subclassing `nn.Module`.

### Basic Components
- `nn.Linear(in_features, out_features)`: A fully connected (dense) layer.
- `nn.ReLU()`: Rectified Linear Unit activation ($max(0, x)$), introduces non-linearity.
- `nn.Sequential()`: A container to stack layers in order.

### Example Model
```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
    
    def forward(self, x):
        return self.layers(x)
```

## 5. The Training Process
Training involves a repetitive loop of 5 essential steps:

1. **`optimizer.zero_grad()`**: Clear previous gradients.
2. **`outputs = model(inputs)`**: Forward pass (make predictions).
3. **`loss = criterion(outputs, targets)`**: Calculate error (e.g., `nn.MSELoss()`).
4. **`loss.backward()`**: Backward pass (compute gradients via backprop).
5. **`optimizer.step()`**: Update weights based on gradients.

### Inference
Use `with torch.no_grad():` during evaluation to disable gradient tracking, saving memory and speed.

## 6. Key Utilities
- `torch.manual_seed(42)`: Ensures reproducibility.
- `DataLoader` & `Dataset`: Efficiently handle batching and shuffling of data.

