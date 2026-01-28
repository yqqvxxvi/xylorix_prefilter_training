# Wood Classification

CNN-based image classification for wood detection and usability assessment. Supports training, inference, and conversion to TFLite for mobile deployment.

## Features

- **CNN Training**: ResNet18 and EfficientNet (B0-B3) architectures
- **Binary Classification**: Single-output with BCEWithLogitsLoss for stable training
- **TFLite Conversion**: PyTorch to TFLite for React Native deployment
- **Batch Inference**: Classify directories of images with adjustable thresholds
- **ROC Curve Generation**: Evaluate models and find optimal thresholds

## Model Performance

Production TFLite models evaluated on independent test sets (January 2026):

### Endgrain Detection Model
**Test Set:** 1,209 images (801 positive, 408 negative)

| Metric | Value |
|--------|-------|
| **Accuracy** | **96.44%** |
| **Precision** | 99.87% |
| **Recall** | 94.76% |
| **F1 Score** | 97.25% |
| **AUC** | 99.89% |
| **False Negative Rate** | 5.24% |

**Model Details:**
- Input: 224×224 grayscale (1 channel)
- Size: 15 MB (float32)
- Location: `models/inuse_27012026_before/endgrain.tflite`

**Performance Highlights:**
- Excellent precision (99.87%) - minimal false positives
- Strong recall (94.76%) - catches most positive cases
- Very low false positive rate (0.25%)

### Usability Classification Model
**Test Set:** 194 images (120 positive, 74 negative)

| Metric | Value |
|--------|-------|
| **Accuracy** | **95.88%** |
| **Precision** | 93.75% |
| **Recall** | 100.00% |
| **F1 Score** | 96.77% |
| **AUC** | 99.80% |
| **False Negative Rate** | 0.00% |

**Model Details:**
- Input: 224×224 RGB (3 channels)
- Size: 15 MB (float32)
- Location: `models/inuse_27012026_before/usability.tflite`

**Performance Highlights:**
- Perfect recall (100%) - no false negatives
- High accuracy (95.88%) on usability assessment
- Conservative on negative predictions (10.81% FPR)

### Evaluation Outputs

Run comprehensive evaluation with:
```bash
python scripts/evaluate_tflite.py --output-dir test_result/
```

Generated outputs include:
- Confusion matrices
- ROC curves with optimal thresholds
- Sample predictions visualization
- False negative/positive analysis
- Sharpness analysis integration
- Detailed metrics (JSON + text reports)

See full evaluation report: `test_result/summary_report.txt`

## Installation

```bash
pip install -r requirements.txt

# For TFLite conversion
pip install -r requirements-tflite.txt
```

## Project Structure

```
wood-classification/
├── scripts/
│   ├── train_cnn.py              # Train CNN models
│   ├── convert_to_tflite.py      # Convert to TFLite
│   ├── evaluate_tflite.py        # TFLite model evaluation
│   ├── batch_classify_wood.py    # Batch wood classification
│   ├── batch_classify_usability.py
│   ├── evaluate_roc.py           # Generate ROC curves
│   └── predict.py                # Single image inference
├── src/
│   ├── data/                     # Data loading and transforms
│   ├── models/                   # Model architectures
│   ├── training/                 # Training utilities
│   └── utils/                    # Metrics and plotting
├── models/
│   └── inuse_27012026_before/    # Production TFLite models
│       ├── endgrain.tflite       # 96.44% accuracy
│       └── usability.tflite      # 95.88% accuracy
├── config/
│   ├── default.yaml              # Default CNN settings
│   └── usability.yaml            # Usability task settings
├── requirements.txt
└── requirements-tflite.txt
```

## Dataset Structure

```
dataset/
├── wood/           # Positive class
│   └── *.jpg
└── non_wood/       # Negative class
    └── *.jpg
```

---

## Scripts Usage

### 1. Train CNN Model

```bash
# Basic training
python scripts/train_cnn.py \
    --positive-dir dataset/wood/ \
    --negative-dir dataset/non_wood/ \
    --model resnet18 \
    --epochs 20

# With EfficientNet and augmentation
python scripts/train_cnn.py \
    --positive-dir dataset/wood/ \
    --negative-dir dataset/non_wood/ \
    --model efficientnet_b0 \
    --epochs 30 \
    --batch-size 32 \
    --stack-augmentations \
    --num-augmentations 3

# Usability task with balanced sampling
python scripts/train_cnn.py \
    --task usability \
    --positive-dir dataset/usable/ \
    --negative-dir dataset/unusable/ \
    --model resnet18 \
    --balanced-sampler

# Grayscale training
python scripts/train_cnn.py \
    --positive-dir dataset/wood/ \
    --negative-dir dataset/non_wood/ \
    --model efficientnet_b0 \
    --grayscale
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | wood | `wood` or `usability` |
| `--model` | resnet18 | `resnet18`, `efficientnet_b0-b3` |
| `--epochs` | 20 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--grayscale` | False | Use grayscale images |
| `--balanced-sampler` | False | Weighted sampling for imbalance |
| `--stack-augmentations` | False | Multiply dataset with augmentation |
| `--num-augmentations` | 2 | Augmented copies per image |
| `--early-stopping` | 10 | Early stopping patience |
| `--wandb` | False | Enable W&B logging |

**Output:** `models/cnn/<task>_<model>_<timestamp>/best_model.pt`

---

### 2. Convert to TFLite

```bash
# Basic conversion
python scripts/convert_to_tflite.py \
    --model models/cnn/best_model.pt \
    --input-shape 1 3 224 224

# With float16 quantization (recommended)
python scripts/convert_to_tflite.py \
    --model models/cnn/best_model.pt \
    --input-shape 1 3 224 224 \
    --quantize float16

# From state_dict checkpoint
python scripts/convert_to_tflite.py \
    --model models/cnn/best_model.pt \
    --input-shape 1 1 224 224 \
    --architecture efficientnet_b0 \
    --num-classes 1 \
    --grayscale
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to PyTorch model |
| `--input-shape` | required | Shape: batch channels H W |
| `--quantize` | dynamic | `none`, `dynamic`, `float16`, `int8` |
| `--architecture` | auto | Model type (if state_dict only) |
| `--grayscale` | False | 1-channel input |
| `--output-dir` | outputs/tflite | Output directory |
| `--output-name` | model | Output filename |

**Output:**
- `outputs/tflite/model.tflite`
- `outputs/tflite/model_info.json`

**React Native:**
```javascript
import { useTensorflowModel } from 'react-native-fast-tflite';
const model = useTensorflowModel(require('./model.tflite'));
// Input: NHWC format [1, 224, 224, 3]
const output = model.run(inputData);
```

---

### 3. Batch Classification

```bash
# Wood classification
python scripts/batch_classify_wood.py \
    --model models/cnn/best_model.pt \
    --input-dir images/ \
    --output-dir results/ \
    --threshold 0.5 \
    --copy-files

# Usability classification
python scripts/batch_classify_usability.py \
    --model models/usability/best_model.pt \
    --input-dir wood_images/ \
    --output-dir results/ \
    --threshold 0.5 \
    --copy-files
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to CNN model |
| `--input-dir` | required | Input image directory |
| `--output-dir` | required | Output directory |
| `--threshold` | 0.5 | Classification threshold |
| `--copy-files` | False | Organize into subdirectories |

**Output:**
- `results/results.csv`
- `results/wood/`, `results/non_wood/` (if --copy-files)

---

### 4. Single Image Prediction

```bash
# Single image
python scripts/predict.py \
    --model models/cnn/best_model.pt \
    --image test.jpg

# Batch
python scripts/predict.py \
    --model models/cnn/best_model.pt \
    --input-dir images/ \
    --output predictions.csv
```

---

### 5. ROC Curve Evaluation

```bash
python scripts/evaluate_roc.py \
    --model_path models/cnn/best_model.pt \
    --positive_dir dataset/test/wood/ \
    --negative_dir dataset/test/non_wood/ \
    --output_path results/roc_curve.png \
    --plot_thresholds
```

**Output:**
- ROC curve plot with AUC score
- Optimal threshold (Youden's index)

---

### 6. TFLite Model Evaluation

Comprehensive evaluation of TFLite models with test data:

```bash
# Evaluate both endgrain and usability models
python scripts/evaluate_tflite.py --output-dir test_result/

# Evaluate single model
python scripts/evaluate_tflite.py \
    --output-dir test_result/ \
    --models endgrain

# With custom threshold
python scripts/evaluate_tflite.py \
    --output-dir test_result/ \
    --threshold 0.48
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | test_result | Output directory |
| `--models` | both | `endgrain`, `usability`, or both |
| `--threshold` | 0.5 | Classification threshold |
| `--include-sharpness` | True | Include sharpness analysis |

**Output Structure:**
```
test_result/
├── summary_report.txt              # Overall summary
├── summary_metrics.json            # Machine-readable metrics
├── endgrain/
│   ├── confusion_matrix.png        # Confusion matrix
│   ├── roc_curve.png               # ROC with AUC
│   ├── sample_predictions.png      # Sample images
│   ├── false_negatives.png         # Error analysis
│   ├── predictions.csv             # All predictions
│   └── metrics.json                # Detailed metrics
├── usability/
│   └── (same structure)
└── sharpness_analysis/
    ├── blur_detection_summary.png
    ├── sharpness_distribution.png
    └── sharpness_metrics.txt
```

**Features:**
- Auto-detects input format (grayscale/RGB) from model
- Computes accuracy, precision, recall, F1, AUC
- Generates confusion matrices and ROC curves
- Visualizes sample predictions and errors
- Integrates sharpness/blur detection analysis
- Exports detailed CSV and JSON reports

---

## Typical Workflow

```bash
# 1. Train
python scripts/train_cnn.py \
    --positive-dir dataset/wood/ \
    --negative-dir dataset/non_wood/ \
    --model efficientnet_b0 \
    --epochs 30

# 2. Evaluate (find optimal threshold)
python scripts/evaluate_roc.py \
    --model_path models/cnn/wood_*/best_model.pt \
    --positive_dir dataset/test/wood/ \
    --negative_dir dataset/test/non_wood/

# 3. Batch classify with optimal threshold
python scripts/batch_classify_wood.py \
    --model models/cnn/wood_*/best_model.pt \
    --input-dir production_images/ \
    --output-dir classified/ \
    --threshold 0.48 \
    --copy-files

# 4. Convert for mobile
python scripts/convert_to_tflite.py \
    --model models/cnn/wood_*/best_model.pt \
    --input-shape 1 3 224 224 \
    --quantize float16
```

---

## Configuration

Edit `config/default.yaml`:

```yaml
cnn:
  model: 'efficientnet_b0'
  pretrained: true
  image_size: 224
  batch_size: 32
  epochs: 20
  learning_rate: 0.001
  use_balanced_sampler: false
  grayscale: false
```

---

## Hardware Support

- **CUDA**: NVIDIA GPUs (auto-detected)
- **MPS**: Apple Silicon M1/M2/M3 (auto-detected)
- **CPU**: Fallback

Override: `--device cuda|mps|cpu`

---

## Model Architecture

Binary classification with single output:
- **Output**: Single value → sigmoid → probability [0, 1]
- **Loss**: BCEWithLogitsLoss
- **Threshold**: Default 0.5, adjustable

Supported models:
- ResNet18
- EfficientNet B0, B1, B2, B3
