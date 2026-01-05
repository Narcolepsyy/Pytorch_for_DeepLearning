# PyTorch Fundamentals: Week 4 Cheat Sheet

## 1. Convolutional Neural Networks (CNNs)
CNNs are designed to process grid-like data, such as images, by using filters to extract spatial features.

### Key Layers
- **`nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`**: Applies a 2D convolution.
    - `in_channels`: Number of input channels (e.g., 3 for RGB).
    - `out_channels`: Number of filters (feature maps) to produce.
    - `kernel_size`: Size of the sliding window (e.g., 3 for a 3x3 filter).
- **`nn.MaxPool2d(kernel_size, stride)`**: Reduces spatial dimensions (downsampling) by taking the maximum value in a window.
- **`nn.Flatten()`**: Converts multi-dimensional feature maps into a 1D vector before passing them to fully connected layers.
- **`nn.Dropout(p)`**: Randomly zeroes out elements with probability `p` during training to prevent overfitting.

### CNN Architecture Example
```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), # Assuming 32x32 input -> 8x8 after two 2x2 pools
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.network(x)
```

## 2. Image Preprocessing & Data Loading
PyTorch uses `torchvision.transforms` to prepare image data.

### Common Transformations
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize images to a fixed size
    transforms.ToTensor(),              # Convert image to tensor [0, 1]
    transforms.Normalize(               # Standardize using mean and std
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

### Loading Image Data
- **`datasets.ImageFolder(root, transform)`**: Loads images from a directory structured by class folders.
- **`DataLoader(dataset, batch_size, shuffle)`**: Handles batching and shuffling.

## 3. Debugging PyTorch Models
Debugging deep learning models often involves tracking tensor shapes and gradients.

### Shape Debugging
Insert print statements in the `forward` method to verify dimensions:
```python
def forward(self, x):
    print(f"Input: {x.shape}")
    x = self.conv1(x)
    print(f"After Conv1: {x.shape}")
    return x
```

### Common Issues
- **Shape Mismatch**: Occurs when the output of one layer doesn't match the expected input of the next (common at the `nn.Linear` transition).
- **Vanishing/Exploding Gradients**: Check if weights or gradients become `NaN` or `Inf`.
- **`torch.autograd.set_detect_anomaly(True)`**: Enable this to find the exact operation that produced a `NaN` during the backward pass.

## 4. Training Best Practices
- **Validation Set**: Always evaluate on a separate validation set to monitor for overfitting.
- **Model Saving/Loading**:
  ```python
  # Save
  torch.save(model.state_dict(), 'model.pth')
  # Load
  model.load_state_dict(torch.load('model.pth'))
  model.eval() # Set to evaluation mode
  ```
