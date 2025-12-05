# Wood Classification

A machine learning and deep learning framework for wood microscopy image classification. This project supports both binary classification tasks:
- **Wood vs Non-Wood**: Classify microscopy images as wood or non-wood
- **Usability**: Classify wood images as usable or unusable

## Recent Updates (December 2025)

### Binary Classification Refactor
- **Clean Binary Classification**: All models now use single output neuron (num_classes=1) with sigmoid activation
- **BCEWithLogitsLoss**: Proper binary classification loss for more stable training
- **Adjustable Thresholds**: Classification threshold can be adjusted without retraining (default: 0.5)
- **Auto-Detection**: Scripts automatically detect model architecture from checkpoints
- **Backward Compatible**: Works with both old (2-output) and new (1-output) models

### ROC Curve Generation
- **Comprehensive ROC Analysis**: Generate ROC curves from two class directories
- **Automatic Batch Processing**: Efficiently processes large datasets
- **Optimal Threshold Detection**: Finds best threshold using Youden's index
- **Threshold Visualization**: Optionally annotate specific threshold points on curves

### Improved Batch Processing
- **Enhanced Confidence Reporting**: Breakdown by class showing high confidence predictions per category
- **Flexible Thresholds**: Separate classification and confidence thresholds
- **Better Statistics**: Clear reporting of prediction distributions

### Previous Updates
- **Augmentation Stacking**: Multiply dataset size by creating multiple augmented versions per image
- **Mirrored Edge Rotation**: Rotation augmentation uses mirrored edges instead of black borders
- **Improved API**: Parameter names changed to `positive_dir`/`negative_dir` for better generalization

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training](#training)
  - [CNN Training](#1-cnn-training)
  - [Machine Learning on Features](#2-machine-learning-on-features)
- [Evaluation](#evaluation)
  - [ROC Curve Generation](#roc-curve-generation)
- [Batch Processing](#batch-processing)
  - [Batch Wood Classification](#batch-wood-classification)
  - [Batch Usability Classification](#batch-usability-classification)
- [Inference](#inference)
- [Binary Classification Guide](#binary-classification-guide)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)

---

## Features

- **Clean Binary Classification**:
  - Single output neuron with sigmoid activation
  - Adjustable classification threshold (default: 0.5)
  - BCEWithLogitsLoss for stable training
  - Backward compatible with old models

- **ROC Curve Analysis**:
  - Generate comprehensive ROC curves
  - Automatic optimal threshold detection
  - Batch processing from directories
  - Beautiful publication-ready plots

- **Enhanced Batch Processing**:
  - Threshold-based classification
  - Detailed confidence reporting by class
  - Automatic file organization
  - Support for CNN, Random Forest, and MLP

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

## Evaluation

### ROC Curve Generation

Generate comprehensive ROC curves to evaluate model performance and find optimal classification thresholds.

#### Basic Usage

```bash
python scripts/evaluate_roc.py \
  --model_path models/best_model.pt \
  --positive_dir data/test/wood \
  --negative_dir data/test/non_wood \
  --model_type efficientnet \
  --output_path results/roc_curve.png
```

#### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--model_path` | str | Yes | - | Path to trained model checkpoint |
| `--positive_dir` | str | Yes | - | Directory with positive class images |
| `--negative_dir` | str | Yes | - | Directory with negative class images |
| `--model_type` | str | No | `resnet18` | Model architecture (`resnet18`, `efficientnet`) |
| `--output_path` | str | No | `results/roc_curve.png` | Path to save ROC curve plot |
| `--batch_size` | int | No | 32 | Batch size for processing |
| `--device` | str | No | Auto | Device (`cuda`, `cpu`) |
| `--image_size` | int | No | 224 | Input image size |
| `--class_names` | str[] | No | `['Non-Wood', 'Wood']` | Class names for legend |
| `--plot_thresholds` | flag | No | - | Annotate threshold values on plot |
| `--threshold_step` | int | No | 10 | Step for threshold annotations |
| `--show` | flag | No | - | Display plot after generation |

#### Features

- **Automatic Batch Processing**: Efficiently processes all images from directories
- **Auto-Detection**: Detects model architecture (1-output vs 2-output) automatically
- **Optimal Threshold**: Finds best threshold using Youden's index (maximizes TPR - FPR)
- **Comprehensive Metrics**: Returns AUC, TPR, FPR, and all threshold points
- **Beautiful Plots**: Publication-ready visualizations with optional threshold annotations

#### Output

The script provides:
- **ROC Curve Plot**: Visual representation with AUC score
- **Optimal Threshold**: Best classification threshold
- **Performance Metrics**: TPR and FPR at optimal threshold
- **Raw Data**: All threshold values and corresponding TPR/FPR rates

#### Example Output

```
AUC: 0.9543
Optimal Threshold: 0.4823
  - True Positive Rate: 0.9123
  - False Positive Rate: 0.0456
```

#### Use Cases

1. **Find Optimal Threshold**: Use the optimal threshold for batch classification
2. **Model Comparison**: Compare multiple models by AUC scores
3. **Performance Analysis**: Understand trade-offs between TPR and FPR
4. **Threshold Selection**: Choose threshold based on your precision/recall requirements

---

## Batch Processing

Process large numbers of images efficiently with automatic classification and organization.

### Batch Wood Classification

Classify images as wood or non-wood with confidence reporting.

#### Basic Usage

```bash
python scripts/batch_classify_wood.py \
  --model-type cnn \
  --model models/wood_model.pt \
  --input-dir data/images/ \
  --output-dir results/
```

#### Key Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-type` | str | Required | Model type (`rf`, `mlp`, `cnn`) |
| `--model` | str | Required | Path to model file |
| `--input-dir` | str | Required | Input directory with images |
| `--output-dir` | str | Required | Output directory for results |
| `--threshold` | float | 0.5 | **Classification threshold** |
| `--confidence-threshold` | float | 0.0 | Minimum confidence for reporting |
| `--copy-files` | flag | - | Copy images to subdirectories |

#### New Threshold Feature

The `--threshold` parameter controls when an image is classified as positive:

```bash
# Default threshold (0.5) - balanced
python scripts/batch_classify_wood.py \
  --model-type cnn \
  --model models/wood_model.pt \
  --input-dir data/test/ \
  --output-dir results/ \
  --threshold 0.5

# Lower threshold (0.3) - more wood detections (higher recall)
python scripts/batch_classify_wood.py \
  --model-type cnn \
  --model models/wood_model.pt \
  --input-dir data/test/ \
  --output-dir results/ \
  --threshold 0.3

# Higher threshold (0.7) - fewer wood detections (higher precision)
python scripts/batch_classify_wood.py \
  --model-type cnn \
  --model models/wood_model.pt \
  --input-dir data/test/ \
  --output-dir results/ \
  --threshold 0.7
```

**Use optimal threshold from ROC curve:**
```bash
# First, find optimal threshold
python scripts/evaluate_roc.py \
  --model_path models/wood_model.pt \
  --positive_dir data/test/wood \
  --negative_dir data/test/non_wood \
  --output_path results/roc.png

# Output: Optimal Threshold: 0.4823

# Then use it for batch processing
python scripts/batch_classify_wood.py \
  --model-type cnn \
  --model models/wood_model.pt \
  --input-dir data/production/ \
  --output-dir results/ \
  --threshold 0.4823  # Use optimal threshold!
```

#### Enhanced Confidence Reporting

The script now provides detailed confidence breakdown:

```
RESULTS SUMMARY
================================================================================
Total images: 100

Prediction counts:
non_wood    60
wood        40

High confidence (>= 0.7): 85 (85.0%)
  - wood: 35 (87.5% of wood predictions)
  - non_wood: 50 (83.3% of non_wood predictions)
```

### Batch Usability Classification

Same features as wood classification, optimized for usability task:

```bash
python scripts/batch_classify_usability.py \
  --model-type cnn \
  --model models/usability_model.pt \
  --input-dir data/wood_images/ \
  --output-dir results/ \
  --threshold 0.5 \
  --confidence-threshold 0.6 \
  --copy-files
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

## Binary Classification Guide

### Understanding the New Architecture

All CNN models now use **proper binary classification** with a single output neuron:

**Old Approach (num_classes=2):**
- Model outputs two values → softmax → argmax
- Threshold fixed at 0.5
- Cannot adjust without retraining

**New Approach (num_classes=1):**
- Model outputs single value → sigmoid → threshold-based decision
- Threshold adjustable (default: 0.5)
- Can optimize threshold without retraining

### How It Works

```python
# Model forward pass
output = model(image)  # Single raw logit value

# Apply sigmoid to get probability
probability = sigmoid(output)  # Value between 0 and 1

# Threshold-based prediction
if probability >= threshold:  # Default threshold = 0.5
    prediction = "positive"  # e.g., wood, usable
else:
    prediction = "negative"  # e.g., non-wood, unusable

# Confidence
confidence = probability if prediction == "positive" else (1 - probability)
```

### Training Details

- **Loss Function**: `BCEWithLogitsLoss` (combines sigmoid + BCE for stability)
- **Output**: Single neuron with no activation (raw logits)
- **Inference**: Apply sigmoid to get probabilities

### Backward Compatibility

All scripts automatically detect model architecture:
- **New models** (1 output): Use sigmoid
- **Old models** (2 outputs): Use softmax

Works seamlessly with both!

### Choosing the Right Threshold

1. **Use Default (0.5)**: Good starting point, balanced predictions
2. **Use ROC Curve**: Find optimal threshold for your data
3. **Adjust for Task**:
   - **High Recall needed**: Lower threshold (0.3-0.4)
   - **High Precision needed**: Higher threshold (0.6-0.7)

### Complete Workflow Example

```bash
# 1. Train a model (automatically uses num_classes=1)
python scripts/train_cnn.py \
  --task wood \
  --model efficientnet_b0 \
  --positive-dir data/wood/ \
  --negative-dir data/non_wood/

# 2. Evaluate and find optimal threshold
python scripts/evaluate_roc.py \
  --model_path models/cnn/wood_*/best_model.pt \
  --positive_dir data/test/wood \
  --negative_dir data/test/non_wood \
  --model_type efficientnet

# Output: Optimal Threshold: 0.4823, AUC: 0.9543

# 3. Batch classify with optimal threshold
python scripts/batch_classify_wood.py \
  --model-type cnn \
  --model models/cnn/wood_*/best_model.pt \
  --input-dir data/production/ \
  --output-dir results/ \
  --threshold 0.4823  # Use optimal threshold from step 2
```

### Key Benefits

✅ **Adjustable Threshold**: Change without retraining
✅ **Cleaner Code**: Standard binary classification pattern
✅ **Better Performance**: More stable training with BCEWithLogitsLoss
✅ **Flexible Deployment**: Optimize threshold for different use cases
✅ **Backward Compatible**: Works with old models too

For detailed technical information, see `docs/BINARY_CLASSIFICATION_CHANGES.md`.

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
