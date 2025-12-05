# Batch Processing Guide

This guide explains how to use the batch processing scripts for large-scale wood classification workflows.

## Table of Contents

- [Overview](#overview)
- [Training Visualization](#training-visualization)
- [Individual Batch Scripts](#individual-batch-scripts)
- [Full Pipeline](#full-pipeline)
- [Examples](#examples)

---

## Overview

The batch processing system provides:

1. **Automated CSV export** of training metrics for better debugging
2. **Standalone plotting** from CSV files for flexible visualization
3. **Individual stage processing** for wood classification, VoL filtering, and usability assessment
4. **Complete pipeline** that chains all stages together
5. **Configurable confidence thresholds** at each stage

### Pipeline Flow

```
Raw Images
    ↓
[Stage 1: Wood vs Non-Wood Classification]
    ↓
Wood Images ────→ Non-Wood Images
    ↓
[Stage 2: VoL Filtering (Blur Detection)]
    ↓
Clear Wood ────→ Blurry Wood
    ↓
[Stage 3: Usability Classification]
    ↓
Usable ────→ Unusable
```

---

## Training Visualization

### Automatic CSV Export

All training scripts now automatically save metrics to CSV:

**CNN Training** (`train_cnn.py`):
- Saves `training_history.csv` with columns: `epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc`

**ML Training** (`train_blob_ml.py`):
- Saves `training_metrics.csv` with columns: `split`, `accuracy`, `precision`, `recall`, `f1`, `auc`

### Plotting Training History

Use `plot_training.py` to generate visualizations from CSV files:

```bash
# Plot from CSV file
python scripts/plot_training.py --csv models/cnn/wood_resnet18_20251205/training_history.csv

# Plot from model directory (auto-finds CSV)
python scripts/plot_training.py --model-dir models/cnn/wood_resnet18_20251205/

# Save to custom location
python scripts/plot_training.py --csv models/cnn/training_history.csv --output my_plot.png

# Show plot interactively
python scripts/plot_training.py --csv models/cnn/training_history.csv --show
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `--csv` | str | Path to training_history.csv file |
| `--model-dir` | str | Path to model directory (auto-finds CSV) |
| `--output` | str | Output path for plot (default: same dir as CSV) |
| `--show` | flag | Show plot interactively |

**Output:**
- Training and validation loss curves
- Training and validation accuracy curves
- Markers for best epochs
- Summary statistics printed to console

---

## Individual Batch Scripts

### 1. Wood vs Non-Wood Classification

**Script:** `batch_classify_wood.py`

Classify a directory of images as wood or non-wood.

```bash
# Using CNN model
python scripts/batch_classify_wood.py \
  --model-type cnn \
  --model models/wood/best_model.pt \
  --input-dir dataset/raw_images/ \
  --output-dir outputs/wood_classified/ \
  --copy-files

# Using Random Forest with texture features
python scripts/batch_classify_wood.py \
  --model-type rf \
  --model models/wood/blob_ml/ \
  --input-dir dataset/raw_images/ \
  --output-dir outputs/wood_classified/ \
  --use-texture \
  --confidence-threshold 0.8 \
  --copy-files
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--model-type` | str | Yes | Model type: `rf`, `mlp`, or `cnn` |
| `--model` | str | Yes | Path to model file or directory |
| `--input-dir` | str | Yes | Input directory with images |
| `--output-dir` | str | Yes | Output directory for results |
| `--csv-output` | str | No | Custom CSV path (default: output_dir/results.csv) |
| `--use-texture` | flag | No | Use texture features (RF/MLP only) |
| `--vol-threshold` | float | No | VoL threshold (default: 900) |
| `--image-size` | int | No | Image size for CNN (default: 224) |
| `--device` | str | No | Device: `cuda`, `mps`, or `cpu` |
| `--copy-files` | flag | No | Copy images to subdirectories |
| `--confidence-threshold` | float | No | Minimum confidence (default: 0.0) |

**Output:**
- `results.csv` with predictions, confidences, and probabilities
- Subdirectories (if `--copy-files`): `wood/`, `non_wood/`, `blurry/`, `error/`

---

### 2. VoL Filtering (Blur Detection)

**Script:** `batch_filter_vol.py`

Filter images based on Variance of Laplacian to detect blurry images.

```bash
python scripts/batch_filter_vol.py \
  --input-dir outputs/wood_classified/wood/ \
  --output-dir outputs/vol_filtered/ \
  --threshold 900 \
  --copy-files
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--input-dir` | str | Yes | Input directory with images |
| `--output-dir` | str | Yes | Output directory for results |
| `--csv-output` | str | No | Custom CSV path (default: output_dir/vol_results.csv) |
| `--threshold` | float | No | VoL threshold (default: 900) |
| `--copy-files` | flag | No | Copy images to subdirectories |

**Output:**
- `vol_results.csv` with VoL scores and clear/blurry status
- Subdirectories (if `--copy-files`): `clear/`, `blurry/`
- Statistics: mean, median, min, max VoL scores

---

### 3. Usability Classification

**Script:** `batch_classify_usability.py`

Classify clear wood images as usable or unusable.

```bash
# Using CNN model
python scripts/batch_classify_usability.py \
  --model-type cnn \
  --model models/usability/best_model.pt \
  --input-dir outputs/vol_filtered/clear/ \
  --output-dir outputs/usability_classified/ \
  --copy-files

# Using MLP with texture features and confidence threshold
python scripts/batch_classify_usability.py \
  --model-type mlp \
  --model models/usability/mlp_model.pt \
  --input-dir outputs/vol_filtered/clear/ \
  --output-dir outputs/usability_classified/ \
  --use-texture \
  --confidence-threshold 0.7 \
  --copy-files
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--model-type` | str | Yes | Model type: `rf`, `mlp`, or `cnn` |
| `--model` | str | Yes | Path to model file or directory |
| `--input-dir` | str | Yes | Input directory with wood images |
| `--output-dir` | str | Yes | Output directory for results |
| `--csv-output` | str | No | Custom CSV path (default: output_dir/results.csv) |
| `--use-texture` | flag | No | Use texture features (RF/MLP only) |
| `--vol-threshold` | float | No | VoL threshold (default: 900) |
| `--skip-vol-check` | flag | No | Skip VoL check (default: True) |
| `--image-size` | int | No | Image size for CNN (default: 224) |
| `--device` | str | No | Device: `cuda`, `mps`, or `cpu` |
| `--copy-files` | flag | No | Copy images to subdirectories |
| `--confidence-threshold` | float | No | Minimum confidence (default: 0.0) |

**Output:**
- `results.csv` with predictions, confidences, and probabilities
- Subdirectories (if `--copy-files`): `usable/`, `unusable/`, `error/`

---

## Full Pipeline

**Script:** `pipeline_full_classification.py`

Run the complete classification pipeline: Wood → VoL → Usability

### Basic Usage

```bash
python scripts/pipeline_full_classification.py \
  --input-dir dataset/raw_images/ \
  --output-dir outputs/pipeline_results/ \
  --wood-model models/wood/best_model.pt \
  --wood-model-type cnn \
  --usability-model models/usability/best_model.pt \
  --usability-model-type cnn
```

### Advanced Usage with Confidence Thresholds

```bash
python scripts/pipeline_full_classification.py \
  --input-dir dataset/raw_images/ \
  --output-dir outputs/pipeline_results/ \
  --wood-model models/wood/best_model.pt \
  --wood-model-type cnn \
  --usability-model models/usability/best_model.pt \
  --usability-model-type cnn \
  --vol-threshold 1000 \
  --wood-confidence-threshold 0.85 \
  --usability-confidence-threshold 0.75 \
  --copy-files \
  --save-intermediates
```

### Mixed Models Example

```bash
# Use Random Forest for wood, CNN for usability
python scripts/pipeline_full_classification.py \
  --input-dir dataset/raw_images/ \
  --output-dir outputs/pipeline_results/ \
  --wood-model models/wood/blob_ml/ \
  --wood-model-type rf \
  --use-texture \
  --usability-model models/usability/best_model.pt \
  --usability-model-type cnn \
  --copy-files
```

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--input-dir` | str | Yes | Input directory with raw images |
| `--output-dir` | str | Yes | Output directory for pipeline results |
| `--wood-model` | str | Yes | Path to wood classification model |
| `--wood-model-type` | str | Yes | Wood model type: `rf`, `mlp`, or `cnn` |
| `--usability-model` | str | Yes | Path to usability model |
| `--usability-model-type` | str | Yes | Usability model type: `rf`, `mlp`, or `cnn` |
| `--vol-threshold` | float | No | VoL threshold (default: 900) |
| `--skip-vol-filtering` | flag | No | Skip VoL filtering stage |
| `--use-texture` | flag | No | Use texture features (RF/MLP) |
| `--wood-confidence-threshold` | float | No | Min confidence for wood (default: 0.0) |
| `--usability-confidence-threshold` | float | No | Min confidence for usability (default: 0.0) |
| `--device` | str | No | Device: `cuda`, `mps`, or `cpu` |
| `--copy-files` | flag | No | Copy images to organized subdirectories |
| `--save-intermediates` | flag | No | Save intermediate results for each stage |

### Output Structure

```
outputs/pipeline_results/
├── stage1_wood_classification/
│   └── results.csv              # (if --save-intermediates)
├── stage2_vol_filtering/
│   └── results.csv              # (if --save-intermediates)
├── stage3_usability_classification/
│   └── results.csv              # (if --save-intermediates)
└── final_results/
    ├── final_results.csv        # Combined results
    ├── usable/                  # (if --copy-files)
    ├── unusable/                # (if --copy-files)
    ├── non_wood/                # (if --copy-files)
    ├── blurry/                  # (if --copy-files)
    ├── low_confidence/          # (if --copy-files)
    └── error/                   # (if --copy-files)
```

---

## Examples

### Example 1: Quick Classification with Default Settings

```bash
# Train CNN model
python scripts/train_cnn.py \
  --task wood \
  --positive-dir dataset/wood/ \
  --negative-dir dataset/non_wood/ \
  --model resnet18 \
  --epochs 20

# Plot training history
python scripts/plot_training.py \
  --model-dir models/cnn/wood_resnet18_*

# Batch classify new images
python scripts/batch_classify_wood.py \
  --model-type cnn \
  --model models/cnn/wood_resnet18_*/best_model.pt \
  --input-dir new_images/ \
  --output-dir results/ \
  --copy-files
```

### Example 2: Full Pipeline with High Confidence Filtering

```bash
# Run complete pipeline with strict thresholds
python scripts/pipeline_full_classification.py \
  --input-dir raw_dataset/ \
  --output-dir pipeline_output/ \
  --wood-model models/wood_cnn/best_model.pt \
  --wood-model-type cnn \
  --usability-model models/usability_cnn/best_model.pt \
  --usability-model-type cnn \
  --wood-confidence-threshold 0.9 \
  --usability-confidence-threshold 0.85 \
  --vol-threshold 1000 \
  --copy-files \
  --save-intermediates
```

### Example 3: Feature-Based Models with Texture

```bash
# Train Random Forest with texture features
python scripts/train_blob_ml.py \
  --task wood \
  --positive-dir dataset/wood/ \
  --negative-dir dataset/non_wood/ \
  --model rf \
  --use-texture \
  --n-estimators 200

# Run pipeline with RF models
python scripts/pipeline_full_classification.py \
  --input-dir raw_images/ \
  --output-dir rf_pipeline_output/ \
  --wood-model models/blob_ml/wood_rf_texture_* \
  --wood-model-type rf \
  --usability-model models/blob_ml/usability_rf_texture_* \
  --usability-model-type rf \
  --use-texture \
  --copy-files
```

### Example 4: Step-by-Step Processing

```bash
# Stage 1: Classify wood vs non-wood
python scripts/batch_classify_wood.py \
  --model-type cnn \
  --model models/wood/best_model.pt \
  --input-dir raw_images/ \
  --output-dir stage1_output/ \
  --copy-files

# Stage 2: Filter by VoL
python scripts/batch_filter_vol.py \
  --input-dir stage1_output/wood/ \
  --output-dir stage2_output/ \
  --threshold 900 \
  --copy-files

# Stage 3: Classify usability
python scripts/batch_classify_usability.py \
  --model-type cnn \
  --model models/usability/best_model.pt \
  --input-dir stage2_output/clear/ \
  --output-dir stage3_output/ \
  --copy-files

# Plot training history for debugging
python scripts/plot_training.py --model-dir models/wood/
python scripts/plot_training.py --model-dir models/usability/
```

### Example 5: Custom VoL Threshold Analysis

```bash
# Test different VoL thresholds
for threshold in 700 900 1100 1300; do
  python scripts/batch_filter_vol.py \
    --input-dir wood_images/ \
    --output-dir vol_analysis/threshold_${threshold}/ \
    --threshold ${threshold} \
    --copy-files
done

# Compare results to choose optimal threshold
```

---

## Tips and Best Practices

### Confidence Thresholds

- **Low confidence (0.5-0.7)**: More inclusive, captures uncertain cases
- **Medium confidence (0.7-0.85)**: Balanced approach, good for most use cases
- **High confidence (0.85-0.95)**: Very selective, ensures high-quality results
- **Very high confidence (>0.95)**: Extremely conservative, may miss valid cases

### VoL Threshold Selection

- **600-800**: Very permissive, includes slightly blurry images
- **900**: Default, good balance for most microscopy images
- **1000-1200**: Stricter, only very clear images
- **>1200**: Very strict, may reject some usable images

Test different thresholds on a sample set before processing large batches.

### Performance Optimization

1. **Use GPU when available**: Specify `--device cuda` for CNN models
2. **Batch size considerations**: Larger batches are faster but use more memory
3. **Parallel processing**: Run multiple instances on different image subsets
4. **Skip intermediate copies**: Omit `--copy-files` if only CSV results are needed

### Debugging

1. **Always use `--save-intermediates`** in pipeline for debugging
2. **Check CSV files** at each stage to identify where issues occur
3. **Use plotting script** to visualize training quality
4. **Start with small batches** to validate configuration before full runs

---

## Troubleshooting

### Common Issues

**Issue: Out of memory errors**
- Reduce batch size (CNN models)
- Use CPU instead of GPU
- Process images in smaller batches

**Issue: All images classified as one class**
- Check model path is correct
- Verify model was trained on correct task
- Check confidence thresholds aren't too strict
- Plot training history to verify model learned

**Issue: VoL filtering removes too many images**
- Lower `--vol-threshold` (try 700-800)
- Check sample images manually to verify they're actually blurry
- Consider if your images have different blur characteristics

**Issue: Low confidence predictions**
- Model may need more training data
- Try different model architecture
- Check if test images are similar to training images
- For feature-based models, try adding `--use-texture`

**Issue: Pipeline runs slowly**
- Use `--device cuda` if GPU available
- Skip `--copy-files` if not needed
- Process subsets in parallel
- For CNN models, images are processed one at a time; this is expected

---

## Monitoring Progress

All scripts show progress bars using `tqdm`. For long-running pipelines:

```bash
# Run in background and save output
python scripts/pipeline_full_classification.py [...args...] > pipeline.log 2>&1 &

# Monitor progress
tail -f pipeline.log
```

---

## Integration with Training

The batch processing scripts are designed to work seamlessly with trained models:

```bash
# 1. Train models (automatically saves metrics to CSV)
python scripts/train_cnn.py --task wood [...args...]
python scripts/train_cnn.py --task usability [...args...]

# 2. Visualize training
python scripts/plot_training.py --model-dir models/cnn/wood_*
python scripts/plot_training.py --model-dir models/cnn/usability_*

# 3. Run batch classification
python scripts/pipeline_full_classification.py [...args...]
```

All training metrics are automatically saved for later analysis!
