# Contrastive Learning for Wood Classification

This module implements self-supervised contrastive learning using SimCLR for wood texture classification.

## Overview

**Contrastive learning** is a self-supervised learning approach that learns representations by comparing different augmented views of the same image. The key idea:
- **Positive pairs** (different augmentations of the same image) should have similar embeddings
- **Negative pairs** (different images) should have different embeddings

This allows the model to learn useful features WITHOUT labels, using only the images themselves!

## Directory Structure

```
src/contrastive/
├── __init__.py              # Module exports
├── augmentations.py         # Aggressive data augmentation for contrastive learning
├── losses.py                # NT-Xent, SimCLR, and SupCon losses
├── model.py                 # SimCLR model (encoder + projection head)
├── dataset.py               # Dataset wrapper for creating positive pairs
├── ANCHOR_GUIDE.md          # Comprehensive guide on anchor selection
└── README.md                # This file

scripts/
└── train_contrastive.py     # Training script
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision tqdm matplotlib
```

### 2. Prepare Your Data

Organize your wood images in the following structure:
```
data/
├── positive/          # Wood images (or class 1)
│   ├── wood_001.jpg
│   ├── wood_002.jpg
│   └── ...
└── negative/          # Non-wood images (or class 0)
    ├── nonwood_001.jpg
    ├── nonwood_002.jpg
    └── ...
```

### 3. Train the Model

```bash
python scripts/train_contrastive.py \
    --data_dir data/ \
    --encoder resnet50 \
    --batch_size 64 \
    --epochs 200 \
    --augmentation_strength strong \
    --output_dir outputs/contrastive
```

### 4. Use the Trained Encoder

After training, use the learned encoder for downstream tasks:

```python
import torch
from src.contrastive import SimCLRModel

# Load trained model
checkpoint = torch.load('outputs/contrastive/best_model.pth')
model = SimCLRModel(encoder_name='resnet50', pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Extract features (without projection head)
model.eval()
with torch.no_grad():
    features = model.get_representation(images)  # (batch_size, 2048)

# Use features for classification, clustering, etc.
```

## Components

### 1. Augmentations (`augmentations.py`)

Implements aggressive data augmentation following SimCLR:

```python
from src.contrastive import get_contrastive_augmentation, ContrastiveTransformations

# Create augmentation pipeline
base_transform = get_contrastive_augmentation(
    image_size=224,
    grayscale=False,
    strength='strong'  # Options: 'weak', 'medium', 'strong'
)

# Wrapper to create multiple views
contrastive_transform = ContrastiveTransformations(base_transform, n_views=2)
```

**Augmentation types:**
- Random resized crop (8-100% of original)
- Random horizontal flip
- Strong color jitter (brightness, contrast, saturation, hue)
- Random grayscale conversion
- Gaussian blur
- Optional: Solarization

### 2. Losses (`losses.py`)

Implements contrastive loss functions:

#### NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)
```python
from src.contrastive import NTXentLoss

criterion = NTXentLoss(temperature=0.5)
loss = criterion(z1, z2)  # z1, z2 are embeddings of positive pairs
```

**Parameters:**
- `temperature`: Controls focus on hard negatives
  - Lower (0.1-0.3): More aggressive, focuses on hardest negatives
  - Medium (0.4-0.6): Balanced (recommended)
  - Higher (0.7-1.0): Softer, considers all negatives equally

#### Supervised Contrastive Loss (Optional)
```python
from src.contrastive import SupConLoss

criterion = SupConLoss(temperature=0.5)
loss = criterion(features, labels)  # Uses label information
```

### 3. Model (`model.py`)

SimCLR model with encoder + projection head:

```python
from src.contrastive import SimCLRModel

model = SimCLRModel(
    encoder_name='resnet50',      # Backbone: resnet18/50/101, efficientnet_b0
    pretrained=False,             # Don't use ImageNet weights for SSL
    projection_dim=128,           # Projection head output dimension
    grayscale=False               # Set True for grayscale images
)

# Forward pass returns both representation and projection
representation, projection = model(images)

# For downstream tasks, use representation only
features = model.get_representation(images)
```

**Encoder options:**
- `resnet18`: Smaller, faster (encoder_dim=512)
- `resnet50`: Balanced (encoder_dim=2048) - **Recommended**
- `resnet101`: Larger, more capacity (encoder_dim=2048)
- `efficientnet_b0`: Efficient (encoder_dim=1280)

### 4. Dataset (`dataset.py`)

Wrapper to create positive pairs:

```python
from src.data.dataset import WoodImageDataset
from src.contrastive import ContrastiveDataset, contrastive_collate_fn
from torch.utils.data import DataLoader

# Create base dataset
base_dataset = WoodImageDataset.from_directories(
    positive_dir='data/positive',
    negative_dir='data/negative',
    transform=None
)

# Wrap with contrastive dataset
train_dataset = ContrastiveDataset(
    base_dataset=base_dataset,
    transform=contrastive_transform,
    n_views=2,
    return_label=False  # Labels not needed for self-supervised
)

# Create dataloader with custom collate function
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=contrastive_collate_fn
)
```

## Training Arguments

### Data Arguments
- `--data_dir`: Root directory containing positive/ and negative/ subdirectories
- `--grayscale`: Use grayscale images (1 channel)

### Model Arguments
- `--encoder`: Encoder architecture (resnet18/50/101, efficientnet_b0)
- `--projection_dim`: Projection head output dimension (default: 128)

### Training Arguments
- `--epochs`: Number of training epochs (default: 100, recommend 200+ for self-supervised)
- `--batch_size`: Batch size (default: 64, larger is better!)
- `--lr`: Learning rate (default: 0.0003)
- `--temperature`: Temperature for NT-Xent loss (default: 0.5)
- `--augmentation_strength`: Augmentation strength (weak/medium/strong, default: strong)

### Other Arguments
- `--output_dir`: Output directory for checkpoints
- `--save_freq`: Save checkpoint every N epochs
- `--num_workers`: Number of data loading workers

## Understanding Anchors

**See [ANCHOR_GUIDE.md](ANCHOR_GUIDE.md) for a comprehensive tutorial on anchor selection!**

Quick summary:
- **Anchor**: The reference sample (automatically handled)
- **Positive**: Different augmentation of the same image
- **Negative**: Augmentations of different images
- You don't manually set anchors - the system handles it automatically!

## Training Tips

### 1. Batch Size
- **Larger is better!** More negatives per anchor → better learning
- Minimum: 32 (for small GPUs)
- Recommended: 64-128 (for medium GPUs)
- Optimal: 256+ (for large GPUs or multi-GPU)
- If GPU memory is limited, use gradient accumulation

### 2. Augmentation Strength
- **Strong**: Best for self-supervised pre-training from scratch (recommended)
- **Medium**: Good balance, use if strong augmentation hurts
- **Weak**: For fine-tuning or very limited data

### 3. Training Duration
- Contrastive learning needs MORE epochs than supervised learning
- Minimum: 100 epochs
- Recommended: 200-300 epochs
- For small datasets (< 1000 images): 300-500 epochs

### 4. Temperature
- Start with default (0.5)
- Lower (0.1-0.3) if model converges too slowly
- Higher (0.7-0.9) if training is unstable

### 5. Hardware Requirements
- Minimum: GPU with 6GB VRAM (batch_size=32, resnet18)
- Recommended: GPU with 12GB+ VRAM (batch_size=64, resnet50)
- For faster training: Multi-GPU setup

## Evaluation

After pre-training, evaluate the learned representations:

### Method 1: Linear Probe (Standard Evaluation)
```python
# 1. Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# 2. Train linear classifier
classifier = nn.Linear(model.encoder_dim, num_classes)
# Train classifier with labeled data...

# 3. Evaluate accuracy
# This tells you how good the learned features are!
```

### Method 2: Fine-tuning
```python
# 1. Load pre-trained encoder
# 2. Add classification head
# 3. Fine-tune entire model on labeled data
# Usually achieves better performance than linear probe
```

### Method 3: Feature Extraction
```python
# Use encoder to extract features
# Then train traditional ML models (Random Forest, SVM, etc.)
features = model.get_representation(images)
```

## Example Workflow

### Step 1: Self-supervised Pre-training
```bash
# Train on ALL available images (no labels needed)
python scripts/train_contrastive.py \
    --data_dir data/all_wood_images/ \
    --encoder resnet50 \
    --batch_size 64 \
    --epochs 200 \
    --augmentation_strength strong
```

### Step 2: Evaluate with Linear Probe
```python
# Load pre-trained encoder
checkpoint = torch.load('outputs/contrastive/best_model.pth')
model = SimCLRModel(encoder_name='resnet50')
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# Add and train linear classifier
# This evaluates representation quality
```

### Step 3: Fine-tune for Classification (Optional)
```python
# Fine-tune entire model on labeled subset
# Usually achieves best performance
```

## Comparison with Supervised Learning

| Aspect | Supervised Learning | Contrastive Learning |
|--------|-------------------|---------------------|
| **Labels Required** | Yes, for all data | No (self-supervised) |
| **Data Efficiency** | Needs many labeled samples | Can use unlabeled data |
| **Training Time** | Faster (50-100 epochs) | Slower (200-500 epochs) |
| **Performance** | Good with enough labels | Competitive, better with limited labels |
| **Generalization** | May overfit to label distribution | Learns more general features |

**When to use contrastive learning:**
- ✅ You have lots of unlabeled images, few labeled
- ✅ You want to learn general-purpose features
- ✅ You plan to use the encoder for multiple downstream tasks
- ✅ You have computational resources for longer training

**When to use supervised learning:**
- ✅ You have plenty of labeled data
- ✅ You want faster training
- ✅ You have a single, specific classification task

## References

1. [SimCLR Paper](https://arxiv.org/abs/2002.05709): "A Simple Framework for Contrastive Learning of Visual Representations"
2. [NT-Xent Loss Explained](https://medium.com/self-supervised-learning/nt-xent-loss-normalized-temperature-scaled-cross-entropy-loss-ea5a1ede7c40)
3. [Contrastive Learning Guide](https://encord.com/blog/guide-to-contrastive-learning/)

## Troubleshooting

### Issue: Loss not decreasing
- Increase batch size (more negatives)
- Lower temperature (0.1-0.3)
- Check augmentation (should be strong)
- Train for more epochs

### Issue: Loss decreasing too fast (possible collapse)
- Increase temperature (0.7-0.9)
- Reduce learning rate
- Add weight decay to optimizer

### Issue: Out of memory
- Reduce batch size
- Use smaller encoder (resnet18)
- Use gradient checkpointing
- Reduce image size

### Issue: Poor downstream performance
- Train for more epochs
- Use stronger augmentation
- Increase projection dimension
- Try different temperature

## License

This implementation is based on the SimCLR paper and is provided for educational and research purposes.
