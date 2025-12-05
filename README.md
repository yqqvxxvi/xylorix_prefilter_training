# Wood Classification

A machine learning and deep learning framework for wood microscopy image classification. This project supports both binary classification tasks:
- **Wood vs Non-Wood**: Classify microscopy images as wood or non-wood
- **Usability**: Classify wood images as usable or unusable

## Recent Updates (05/12/2025)

- **Augmentation Stacking**: New feature to multiply dataset size by creating multiple augmented versions per image
- **Mirrored Edge Rotation**: Rotation augmentation now uses mirrored edges instead of black borders for more natural-looking augmented images
- **Improved API**: Parameter names changed from `wood_dir`/`non_wood_dir` to `positive_dir`/`negative_dir` for better generalization
- **Enhanced Data Pipeline**: More flexible dataset creation with `AugmentedWoodDataset` wrapper class

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training](#training)
  - [CNN Training](#1-cnn-training)
  - [Machine Learning on Features](#2-machine-learning-on-features)
- [Inference](#inference)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)

---

## Features

- **Multiple Model Architectures**:
  - Deep Learning: ResNet18, EfficientNet (B0-B3)
  - Classical ML: Random Forest, MLP on hand-crafted features

- **Feature Engineering**:
  - Blob detection and analysis
  - Texture features (LBP, Gabor, GLCM, FFT)
  - Image quality assessment (blur detection via Variance of Laplacian)

- **Flexible Training**:
  - Customizable hyperparameters
  - Transfer learning with pretrained weights
  - Early stopping and learning rate scheduling
  - Class imbalance handling
  - Augmentation stacking to multiply dataset size

- **Advanced Data Augmentation**:
  - Random flips (horizontal and vertical)
  - Random rotation with mirrored edge filling (no black borders)
  - Color jitter for brightness, contrast, and saturation
  - Optional augmentation stacking to create multiple versions per image

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or navigate to the repository:
```bash
cd wood-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install the package in development mode:
```bash
pip install -e .
```

This enables command-line tools:
- `wood-train-cnn` - Train CNN models
- `wood-train-ml` - Train ML models on features
- `wood-predict` - Run inference

---

## Project Structure

```
wood-classification/
├── config/              # Configuration files
│   ├── default.yaml     # Default hyperparameters
│   └── usability.yaml   # Usability task configuration
├── scripts/             # Training and inference scripts
│   ├── train_cnn.py     # CNN training
│   ├── train_blob_ml.py # ML training on features
│   └── predict.py       # Inference script
├── src/                 # Source code
│   ├── data/            # Data loaders and transforms
│   ├── models/          # Model architectures
│   ├── features/        # Feature extraction
│   ├── training/        # Training utilities
│   └── utils/           # Logging and metrics
├── models/              # Saved model checkpoints
├── logs/                # Training logs
└── outputs/             # Output files
```

---

## Quick Start

### Example 1: Train ResNet18 for Wood Classification

```bash
python scripts/train_cnn.py \
  --task wood \
  --positive-dir dataset/wood/ \
  --negative-dir dataset/non_wood/ \
  --model resnet18 \
  --epochs 20 \
  --batch-size 32
```

### Example 2: Train Random Forest on Blob Features

```bash
python scripts/train_blob_ml.py \
  --task wood \
  --positive-dir dataset/wood/ \
  --negative-dir dataset/non_wood/ \
  --model rf \
  --n-estimators 100
```

### Example 3: Run Inference

```bash
python scripts/predict.py \
  --model-type cnn \
  --model models/cnn/best_model.pt \
  --image test_image.jpg
```

---

## Training

### 1. CNN Training

Train deep learning models (ResNet, EfficientNet) using `scripts/train_cnn.py`.

#### Basic Usage

```bash
python scripts/train_cnn.py --positive-dir <path> --negative-dir <path> [OPTIONS]
```

#### Arguments

##### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--positive-dir` | str | Path to positive class images (wood or usable) |
| `--negative-dir` | str | Path to negative class images (non-wood or unusable) |

##### Task Selection

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--task` | str | `wood` | `wood`, `usability` | Classification task type |

##### Model Configuration

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--model` | str | `resnet18` | `resnet18`, `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3` | Model architecture |
| `--pretrained` | flag | `True` | - | Use pretrained ImageNet weights |
| `--no-pretrained` | flag | - | - | Disable pretrained weights |

##### Training Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | `20` | Number of training epochs |
| `--batch-size` | int | `32` | Batch size for training |
| `--lr` | float | `0.001` | Initial learning rate |
| `--weight-decay` | float | `1e-4` | L2 regularization weight decay |
| `--val-split` | float | `0.2` | Validation set fraction (0.0-1.0) |
| `--early-stopping` | int | `10` | Early stopping patience (epochs) |

##### Data Augmentation

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image-size` | int | `224` | Input image size (square) |
| `--balanced-sampler` | flag | - | Use weighted sampling for class imbalance |
| `--stack-augmentations` | flag | - | Stack augmented images on top of originals to increase dataset size |
| `--num-augmentations` | int | `2` | Number of augmented versions per image (when stacking enabled) |
| `--include-original` | flag | `True` | Include original images when stacking (default) |
| `--no-original` | flag | - | Exclude original images, only use augmented versions |

**Data Augmentation Techniques:**
- Random horizontal and vertical flips
- Random rotation (0-360°) with mirrored edge filling
- Color jitter (brightness, contrast, saturation)
- All augmentations applied randomly during training

##### Hardware Settings

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--num-workers` | int | `4` | - | Number of data loading workers |
| `--device` | str | Auto-detect | `cuda`, `mps`, `cpu` | Device for training |

##### Output Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | str | `models/cnn` | Directory to save trained model |
| `--log-dir` | str | `logs` | Directory for training logs |

#### Examples

**Basic training with ResNet18:**
```bash
python scripts/train_cnn.py \
  --task wood \
  --positive-dir dataset/wood/ \
  --negative-dir dataset/non_wood/ \
  --model resnet18
```

**Advanced training with EfficientNet:**
```bash
python scripts/train_cnn.py \
  --task usability \
  --positive-dir dataset/usable/ \
  --negative-dir dataset/unusable/ \
  --model efficientnet_b2 \
  --epochs 50 \
  --batch-size 16 \
  --lr 0.0001 \
  --image-size 256 \
  --balanced-sampler \
  --early-stopping 15
```

**Training without pretrained weights:**
```bash
python scripts/train_cnn.py \
  --positive-dir dataset/wood/ \
  --negative-dir dataset/non_wood/ \
  --model resnet18 \
  --no-pretrained \
  --epochs 100
```

**GPU training:**
```bash
python scripts/train_cnn.py \
  --positive-dir dataset/wood/ \
  --negative-dir dataset/non_wood/ \
  --model efficientnet_b1 \
  --device cuda \
  --batch-size 64 \
  --num-workers 8
```

**Training with augmentation stacking (2x dataset):**
```bash
python scripts/train_cnn.py \
  --positive-dir dataset/wood/ \
  --negative-dir dataset/non_wood/ \
  --model resnet18 \
  --stack-augmentations \
  --num-augmentations 1 \
  --epochs 20
```

**Training with augmentation stacking (4x dataset):**
```bash
python scripts/train_cnn.py \
  --task usability \
  --positive-dir /path/to/usable/ \
  --negative-dir /path/to/unusable/ \
  --model efficientnet_b0 \
  --stack-augmentations \
  --num-augmentations 3 \
  --epochs 30 \
  --balanced-sampler
```

---

### 2. Machine Learning on Features

Train classical ML models (Random Forest, MLP) on hand-crafted blob and texture features using `scripts/train_blob_ml.py`.

#### Basic Usage

```bash
python scripts/train_blob_ml.py --positive-dir <path> --negative-dir <path> [OPTIONS]
```

#### Arguments

##### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--positive-dir` | str | Path to positive class images (wood or usable) |
| `--negative-dir` | str | Path to negative class images (non-wood or unusable) |

##### Task Selection

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--task` | str | `wood` | `wood`, `usability` | Classification task type |

##### Model Selection

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--model` | str | `rf` | `rf`, `mlp` | Model type (Random Forest or MLP) |

##### Feature Extraction

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use-texture` | flag | - | Extract texture features (LBP, Gabor, GLCM, FFT) in addition to blob features |
| `--vol-threshold` | float | `900` | Variance of Laplacian threshold for blur detection |

##### Training Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--test-size` | float | `0.2` | Test set fraction (0.0-1.0) |
| `--random-seed` | int | `42` | Random seed for reproducibility |

##### Random Forest Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n-estimators` | int | `100` | Number of trees in the forest |
| `--max-depth` | int | `None` | Maximum tree depth (None = unlimited) |

##### MLP Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | `100` | Number of training epochs |
| `--batch-size` | int | `32` | Batch size for training |
| `--lr` | float | `0.001` | Learning rate |

##### Output Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | str | `models/blob_ml` | Directory to save trained model |

#### Examples

**Train Random Forest on blob features:**
```bash
python scripts/train_blob_ml.py \
  --task wood \
  --positive-dir dataset/wood/ \
  --negative-dir dataset/non_wood/ \
  --model rf \
  --n-estimators 100
```

**Train Random Forest with texture features:**
```bash
python scripts/train_blob_ml.py \
  --task wood \
  --positive-dir dataset/wood/ \
  --negative-dir dataset/non_wood/ \
  --model rf \
  --use-texture \
  --n-estimators 200 \
  --max-depth 20
```

**Train MLP on blob features:**
```bash
python scripts/train_blob_ml.py \
  --task usability \
  --positive-dir dataset/usable/ \
  --negative-dir dataset/unusable/ \
  --model mlp \
  --epochs 150 \
  --batch-size 64 \
  --lr 0.0005
```

**Train MLP with texture features:**
```bash
python scripts/train_blob_ml.py \
  --positive-dir dataset/wood/ \
  --negative-dir dataset/non_wood/ \
  --model mlp \
  --use-texture \
  --epochs 200 \
  --lr 0.001
```

**Custom blur threshold:**
```bash
python scripts/train_blob_ml.py \
  --positive-dir dataset/wood/ \
  --negative-dir dataset/non_wood/ \
  --model rf \
  --vol-threshold 1200 \
  --use-texture
```

---

## Inference

Run predictions on single images or batches using `scripts/predict.py`.

### Basic Usage

```bash
# Single image
python scripts/predict.py --model-type <type> --model <path> --image <image_path>

# Batch prediction
python scripts/predict.py --model-type <type> --model <path> --input-dir <dir> --output <csv_path>
```

### Arguments

| Argument | Type | Required | Choices | Description |
|----------|------|----------|---------|-------------|
| `--model-type` | str | Yes | `rf`, `mlp`, `cnn` | Type of model to use |
| `--model` | str | Yes | - | Path to model file or directory |
| `--image` | str | Either this or `--input-dir` | - | Single image file path |
| `--input-dir` | str | Either this or `--image` | - | Directory of images for batch prediction |
| `--output` | str | No | - | Output CSV file (for batch prediction) |
| `--use-texture` | flag | No | - | Use texture features (for RF/MLP only) |
| `--vol-threshold` | float | No (default: 900) | - | VoL threshold for blur detection |
| `--image-size` | int | No (default: 224) | - | Image size for CNN |
| `--device` | str | No (auto-detect) | `cuda`, `mps`, `cpu` | Device to use |

### Examples

**Single image prediction with CNN:**
```bash
python scripts/predict.py \
  --model-type cnn \
  --model models/cnn/best_model.pt \
  --image test.jpg
```

**Batch prediction with Random Forest:**
```bash
python scripts/predict.py \
  --model-type rf \
  --model models/blob_ml/model/ \
  --input-dir test_images/ \
  --output results.csv \
  --use-texture
```

**Single image prediction with MLP:**
```bash
python scripts/predict.py \
  --model-type mlp \
  --model models/blob_ml/mlp_model.pt \
  --image sample.jpg \
  --use-texture
```

---

## Configuration

The `config/` directory contains YAML configuration files with default hyperparameters.

### default.yaml

Default configuration for wood vs non-wood classification. Contains settings for:
- Feature extraction parameters
- Random Forest hyperparameters
- MLP hyperparameters
- CNN hyperparameters
- Training configuration

### usability.yaml

Optimized configuration for usability classification with:
- Increased Random Forest estimators (150 trees)
- Deeper MLP architecture (128-64 hidden units)
- Balanced sampling enabled for class imbalance
- Extended training epochs

You can reference these files when setting hyperparameters or modify them for your specific needs.

---

## Model Architectures

### Deep Learning Models

**ResNet18**
- 18-layer residual network
- Pretrained on ImageNet (optional)
- Input: 224x224 RGB images
- Output: Binary classification (wood/non-wood or usable/unusable)

**EfficientNet (B0-B3)**
- Compound scaling architecture
- More efficient than ResNet
- Variants: B0 (smallest) to B3 (larger, more accurate)
- Pretrained on ImageNet (optional)
- Input: 224x224 RGB images

**Data Augmentation Pipeline**
- Images are augmented on-the-fly during training
- Rotation uses mirrored edge filling to avoid black borders
- Augmentation stacking creates multiple augmented versions per image
- Validation set uses only center crop and normalization (no augmentation)

### Feature-Based Models

**Blob Features** (default)
- Cell detection using adaptive thresholding
- Blob statistics: count, area, perimeter, circularity, solidity
- Image quality: Variance of Laplacian (VoL) for blur detection

**Texture Features** (optional with `--use-texture`)
- Local Binary Patterns (LBP)
- Gabor filters
- Gray-Level Co-occurrence Matrix (GLCM)
- Fast Fourier Transform (FFT) analysis

**Random Forest**
- Ensemble of decision trees
- Default: 100 estimators
- No feature scaling required
- Provides feature importance rankings

**MLP (Multi-Layer Perceptron)**
- Architecture: Input → 64 → 32 → Output
- Dropout: 0.5 for regularization
- Features scaled using StandardScaler
- Binary Cross-Entropy loss

---

## Tips and Best Practices

### Data Organization

Organize your dataset with positive and negative classes in separate directories:

```
dataset/
├── wood/              # Positive class
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── non_wood/          # Negative class
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Model Selection

- **CNN models** (ResNet, EfficientNet):
  - Best for raw image classification
  - Requires more data (hundreds to thousands of images)
  - Longer training time
  - Better generalization on complex patterns

- **Feature-based models** (RF, MLP):
  - Good for smaller datasets
  - Faster training
  - Interpretable features
  - Add `--use-texture` for richer feature representation

### Hyperparameter Tuning

**For CNNs:**
- Start with pretrained weights (`--pretrained`)
- Use smaller learning rates (0.0001-0.001) with pretrained models
- Enable `--balanced-sampler` for imbalanced datasets
- Increase `--early-stopping` patience for larger datasets
- Use `--stack-augmentations` to increase effective dataset size:
  - `--num-augmentations 1`: 2x dataset size (1 original + 1 augmented)
  - `--num-augmentations 2`: 3x dataset size (1 original + 2 augmented)
  - `--num-augmentations 3`: 4x dataset size (1 original + 3 augmented)
  - Particularly useful for small datasets (< 500 images per class)

**For Random Forest:**
- Increase `--n-estimators` (100-200) for better performance
- Set `--max-depth` (10-30) to prevent overfitting on small datasets
- Always use `--use-texture` for better accuracy

**For MLP:**
- Scale features (done automatically)
- Use `--use-texture` for more input features
- Tune `--epochs` and `--lr` based on convergence

### Handling Class Imbalance

- Use `--balanced-sampler` for CNN training
- Random Forest handles imbalance naturally
- MLP may benefit from longer training with smaller learning rate
- Combine `--balanced-sampler` with `--stack-augmentations` for best results on imbalanced datasets

### When to Use Augmentation Stacking

**Use `--stack-augmentations` when:**
- You have a small dataset (< 500 images per class)
- Model is overfitting (high training accuracy, low validation accuracy)
- You want to improve model generalization
- You have computational resources for longer training

**Don't use `--stack-augmentations` when:**
- You already have a large dataset (> 2000 images per class)
- Training is already very slow
- You're doing quick experiments and need fast iteration
- Validation accuracy is already good without it

### Output Management

Models are automatically saved with timestamps:
```
models/cnn/wood_resnet18_batch32_lr0.001_20251205_143022/
├── best_model.pt          # Best model checkpoint
├── final_model.pt         # Final model
└── training_history.json  # Training metrics
```

---

## Troubleshooting

**Out of Memory Error**
- Reduce `--batch-size`
- Use smaller model (resnet18 instead of efficientnet_b3)
- Reduce `--image-size`
- Reduce `--num-workers`

**Poor Performance**
- Try enabling `--use-texture` for feature-based models
- Enable `--balanced-sampler` for imbalanced datasets
- Increase training epochs
- Try different model architectures
- Check for data quality issues (blur, artifacts)

**Slow Training**
- Increase `--batch-size` if memory allows
- Increase `--num-workers`
- Use GPU if available (`--device cuda`)
- Reduce `--image-size` for CNN models

**All Images Classified as Blurry**
- Lower `--vol-threshold` (try 600-800)
- Images may genuinely be blurry - check manually

---

## License

MIT License

## Contact

For questions or issues, please open an issue on the project repository.
