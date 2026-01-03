# Data Management: Week 3 Cheat Sheet

## 1. Custom Datasets
When data isn't in a standard format (e.g., images in folders, labels in a CSV), you must subclass `torch.utils.data.Dataset`.

### The 3 Essential Methods
1. **`__init__(self, ...)`**: Initialize data paths, load labels (CSV/JSON), and define transforms.
2. **`__len__(self)`**: Return the total number of samples (`len(data)`).
3. **`__getitem__(self, index)`**: 
   - Load the image (e.g., using `PIL.Image.open`).
   - Apply transforms.
   - Return a tuple: `(image_tensor, label)`.

```python
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.df.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        return image, label
```

## 2. Data Transformations & Augmentation
Transforms prepare data for the model and help prevent overfitting by adding variety.

### Common Transforms
- **`transforms.Resize((h, w))`**: Ensures all images have the same dimensions.
- **`transforms.CenterCrop(size)`**: Crops the center of the image.
- **`transforms.RandomHorizontalFlip(p=0.5)`**: Randomly flips images (Augmentation).
- **`transforms.RandomRotation(degrees)`**: Rotates images (Augmentation).
- **`transforms.Normalize(mean, std)`**: Scales pixel values using dataset-specific statistics.

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 3. Data Splitting
Divide your dataset into **Training**, **Validation**, and **Test** sets.
- **Training**: Used to update model weights.
- **Validation**: Used to tune hyperparameters and detect overfitting.
- **Test**: Used for final evaluation on unseen data.

```python
from torch.utils.data import random_split

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
```

## 4. Efficient Loading with DataLoaders
The `DataLoader` wraps a dataset and provides an iterable over batches.
- **`batch_size`**: Number of samples per iteration.
- **`shuffle=True`**: Shuffles data every epoch (crucial for training).
- **`num_workers`**: Number of CPU processes for data loading (speeds up training).

```python
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
```

## 5. Handling Real-World Issues
- **Corrupted Files**: Use `try-except` blocks inside `__getitem__` to skip or handle broken images.
- **Memory Management**: Only load images in `__getitem__` (on-demand) rather than loading the entire dataset into RAM at once.
- **Class Mappings**: Maintain a dictionary mapping integer labels back to human-readable names (e.g., `{0: 'Aloe Vera', 1: 'Banana'}`).
