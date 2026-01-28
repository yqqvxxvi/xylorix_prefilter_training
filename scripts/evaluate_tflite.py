#!/usr/bin/env python3
"""
TFLite Model Evaluation Script

Runs inference on TFLite models (endgrain and usability) using test data
and generates comprehensive evaluation metrics including:
- Accuracy, Precision, Recall, F1, AUC
- Confusion matrices
- ROC curves
- False negative analysis
- Sample prediction visualizations
- Sharpness analysis integration
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

MODEL_CONFIG = {
    'endgrain': {
        'model_path': 'models/inuse_27012026_before/endgrain.tflite',
        'test_pos': '/Users/youqing/Documents/test_data/endgrain/pos/',
        'test_neg': '/Users/youqing/Documents/test_data/endgrain/neg/',
        'labels': ['Non-Endgrain', 'Endgrain'],
    },
    'usability': {
        'model_path': 'models/inuse_27012026_before/usability.tflite',
        'test_pos': '/Users/youqing/Documents/test_data/usability/pos/',
        'test_neg': '/Users/youqing/Documents/test_data/usability/neg/',
        'labels': ['Not Usable', 'Usable'],
    }
}

PREFILTERED_PIPELINE_PATH = '/Users/youqing/Documents/prefiltered_pipeline_proposal/'

# ImageNet normalization for RGB
RGB_MEAN = np.array([0.485, 0.456, 0.406])
RGB_STD = np.array([0.229, 0.224, 0.225])

# Grayscale normalization
GRAY_MEAN = 0.5
GRAY_STD = 0.5


# ============================================================================
# TFLite Inference Functions
# ============================================================================

def load_tflite_model(model_path: str) -> Tuple[tf.lite.Interpreter, dict, dict]:
    """
    Load TFLite model and return interpreter with input/output details.

    Args:
        model_path: Path to .tflite file

    Returns:
        (interpreter, input_details, output_details)
    """
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    return interpreter, input_details, output_details


def get_image_files(directory: str) -> List[Path]:
    """Get all image files from a directory."""
    directory = Path(directory)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))

    return sorted(files)


def preprocess_image(image_path: str, input_shape: tuple) -> np.ndarray:
    """
    Preprocess image for TFLite inference.
    Auto-detects grayscale vs RGB based on input shape.

    Args:
        image_path: Path to image file
        input_shape: Model input shape (batch, height, width, channels)

    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Get expected dimensions
    _, height, width, channels = input_shape

    # Resize
    image = cv2.resize(image, (width, height))

    if channels == 1:
        # Grayscale preprocessing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) / 255.0
        image = (image - GRAY_MEAN) / GRAY_STD
        image = np.expand_dims(image, axis=(0, -1))  # (1, H, W, 1)
    else:
        # RGB preprocessing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = (image - RGB_MEAN) / RGB_STD
        image = np.expand_dims(image, axis=0)  # (1, H, W, 3)

    return image.astype(np.float32)


def run_inference(interpreter: tf.lite.Interpreter,
                  input_details: dict,
                  output_details: dict,
                  input_data: np.ndarray) -> float:
    """
    Run inference on a single image.

    Args:
        interpreter: TFLite interpreter
        input_details: Input tensor details
        output_details: Output tensor details
        input_data: Preprocessed image

    Returns:
        Probability of positive class
    """
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])

    # Apply sigmoid if output is logit (single value)
    if output.shape[-1] == 1:
        prob = 1 / (1 + np.exp(-output[0, 0]))  # Sigmoid
    else:
        prob = output[0, 1]  # Already softmax, take positive class

    return float(prob)


def batch_evaluate(model_path: str,
                   pos_dir: str,
                   neg_dir: str,
                   threshold: float = 0.5) -> Dict:
    """
    Run inference on all test images and collect results.

    Args:
        model_path: Path to TFLite model
        pos_dir: Directory with positive samples
        neg_dir: Directory with negative samples
        threshold: Classification threshold

    Returns:
        Dictionary with evaluation results
    """
    # Load model
    interpreter, input_details, output_details = load_tflite_model(model_path)
    input_shape = input_details['shape']

    print(f"  Model input shape: {input_shape}")
    print(f"  Input dtype: {input_details['dtype']}")

    # Get image files
    pos_files = get_image_files(pos_dir)
    neg_files = get_image_files(neg_dir)

    print(f"  Positive samples: {len(pos_files)}")
    print(f"  Negative samples: {len(neg_files)}")

    # Collect predictions
    all_paths = []
    all_true = []
    all_probs = []

    # Process positive samples
    print("  Processing positive samples...")
    for img_path in tqdm(pos_files, desc="    Positive"):
        try:
            input_data = preprocess_image(img_path, input_shape)
            prob = run_inference(interpreter, input_details, output_details, input_data)
            all_paths.append(str(img_path))
            all_true.append(1)
            all_probs.append(prob)
        except Exception as e:
            print(f"    Error processing {img_path.name}: {e}")

    # Process negative samples
    print("  Processing negative samples...")
    for img_path in tqdm(neg_files, desc="    Negative"):
        try:
            input_data = preprocess_image(img_path, input_shape)
            prob = run_inference(interpreter, input_details, output_details, input_data)
            all_paths.append(str(img_path))
            all_true.append(0)
            all_probs.append(prob)
        except Exception as e:
            print(f"    Error processing {img_path.name}: {e}")

    # Convert to arrays
    y_true = np.array(all_true)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        'paths': all_paths,
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'input_shape': input_shape.tolist(),
        'threshold': threshold
    }


# ============================================================================
# Metrics Functions
# ============================================================================

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict:
    """
    Compute comprehensive evaluation metrics.

    Returns:
        Dictionary with all metrics
    """
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Rates
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'auc': float(auc),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'false_negative_rate': float(fnr),
        'false_positive_rate': float(fpr),
        'total_samples': int(len(y_true)),
        'positive_samples': int(np.sum(y_true)),
        'negative_samples': int(len(y_true) - np.sum(y_true))
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str],
                          save_path: str,
                          model_name: str):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   save_path: str,
                   model_name: str):
    """Generate and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], c='red', s=100,
                zorder=5, label=f'Optimal (thresh={optimal_threshold:.3f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return optimal_threshold


def plot_sample_predictions(paths: List[str],
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            y_prob: np.ndarray,
                            class_names: List[str],
                            save_path: str,
                            n_samples: int = 16):
    """Create grid showing sample images with predictions."""
    # Select random samples
    indices = np.random.choice(len(paths), min(n_samples, len(paths)), replace=False)

    # Calculate grid size
    n_cols = 4
    n_rows = (len(indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(indices):
            idx = indices[i]
            img = cv2.imread(paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

            ax.imshow(img)

            true_label = class_names[y_true[idx]]
            pred_label = class_names[y_pred[idx]]
            prob = y_prob[idx]

            correct = y_true[idx] == y_pred[idx]
            color = 'green' if correct else 'red'

            ax.set_title(f'True: {true_label}\nPred: {pred_label} ({prob:.2f})',
                        fontsize=9, color=color)
        ax.axis('off')

    plt.suptitle('Sample Predictions', fontsize=14)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_false_negatives(paths: List[str],
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_prob: np.ndarray,
                         class_names: List[str],
                         save_path: str,
                         max_samples: int = 20):
    """Create grid showing false negative examples."""
    # Find false negatives (true=1, pred=0)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]

    if len(fn_indices) == 0:
        print("  No false negatives found!")
        return

    # Limit samples
    n_samples = min(len(fn_indices), max_samples)
    selected_indices = fn_indices[:n_samples]

    # Calculate grid size
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n_samples:
            idx = selected_indices[i]
            img = cv2.imread(paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

            ax.imshow(img)

            prob = y_prob[idx]
            filename = Path(paths[idx]).name

            ax.set_title(f'{filename[:20]}...\nProb: {prob:.3f}', fontsize=8, color='red')
        ax.axis('off')

    plt.suptitle(f'False Negatives ({len(fn_indices)} total, showing {n_samples})', fontsize=14)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Sharpness Analysis Functions
# ============================================================================

def integrate_sharpness_analysis(prefiltered_path: str, output_dir: str):
    """Load and visualize sharpness analysis from prefiltered_pipeline_proposal."""
    prefiltered_path = Path(prefiltered_path)
    output_dir = Path(output_dir) / 'sharpness_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("SHARPNESS ANALYSIS INTEGRATION")
    print("="*60)

    # Load blur detection comparison data
    comparison_csv = prefiltered_path / 'blur_detection_output' / 'blur_detection_comparison.csv'
    if comparison_csv.exists():
        df = pd.read_csv(comparison_csv)
        print(f"  Loaded blur detection comparison: {len(df)} samples")

        # Plot sharpness score distributions
        plot_sharpness_distribution(df, output_dir / 'sharpness_distribution.png')
    else:
        print(f"  Warning: {comparison_csv} not found")
        df = None

    # Load top configurations
    top_configs_csv = prefiltered_path / 'blur_detection_output' / 'top_10_configurations.csv'
    if top_configs_csv.exists():
        top_configs = pd.read_csv(top_configs_csv)
        print(f"  Loaded top configurations: {len(top_configs)} configs")

        # Plot configuration performance
        plot_blur_detection_performance(top_configs, output_dir / 'blur_detection_summary.png')
    else:
        print(f"  Warning: {top_configs_csv} not found")
        top_configs = None

    # Generate text summary
    generate_sharpness_summary(df, top_configs, output_dir / 'sharpness_metrics.txt')

    return df, top_configs


def plot_sharpness_distribution(df: pd.DataFrame, save_path: str):
    """Plot distribution of sharpness scores."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['vol_score', 'fft_score', 'sobel_score']
    titles = ['Variance of Laplacian (VoL)', 'FFT Score', 'Sobel Variance']

    for ax, metric, title in zip(axes, metrics, titles):
        if metric in df.columns:
            # Group by true label
            for label in df['true_label'].unique():
                subset = df[df['true_label'] == label][metric]
                ax.hist(subset, bins=20, alpha=0.6, label=str(label))

            ax.set_xlabel(title)
            ax.set_ylabel('Count')
            ax.set_title(f'{title} Distribution')
            ax.legend()

    plt.suptitle('Sharpness Metric Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_blur_detection_performance(top_configs: pd.DataFrame, save_path: str):
    """Visualize blur detection performance metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of metrics for best configuration
    if len(top_configs) > 0:
        best = top_configs.iloc[0]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [best[m] for m in metrics if m in best.index]

        axes[0].bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel('Score')
        axes[0].set_title('Best Configuration Performance')

        for i, v in enumerate(values):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center')

    # Top 10 configurations comparison
    if len(top_configs) > 1:
        x = range(len(top_configs))
        axes[1].bar(x, top_configs['f1_score'], alpha=0.7)
        axes[1].set_xlabel('Configuration Rank')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Top 10 Configurations by F1 Score')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'#{i+1}' for i in x])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_sharpness_summary(comparison_df: Optional[pd.DataFrame],
                                top_configs: Optional[pd.DataFrame],
                                save_path: str):
    """Generate text summary of sharpness analysis."""
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SHARPNESS / BLUR DETECTION ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        if top_configs is not None and len(top_configs) > 0:
            best = top_configs.iloc[0]
            f.write("BEST BLUR DETECTION CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Accuracy:  {best.get('accuracy', 'N/A'):.4f}\n")
            f.write(f"  Precision: {best.get('precision', 'N/A'):.4f}\n")
            f.write(f"  Recall:    {best.get('recall', 'N/A'):.4f}\n")
            f.write(f"  F1 Score:  {best.get('f1_score', 'N/A'):.4f}\n")
            f.write("\n")
            f.write("Parameters:\n")
            if 'vol_weight' in best.index:
                f.write(f"  VoL Weight:   {best['vol_weight']}\n")
                f.write(f"  FFT Weight:   {best['fft_weight']}\n")
                f.write(f"  Sobel Weight: {best['sobel_weight']}\n")
                f.write(f"  Threshold:    {best['threshold']}\n")

        if comparison_df is not None:
            f.write("\n\nTEST DATA STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total samples: {len(comparison_df)}\n")

            for label in comparison_df['true_label'].unique():
                subset = comparison_df[comparison_df['true_label'] == label]
                f.write(f"\n  {label.upper()} samples ({len(subset)}):\n")

                for metric in ['vol_score', 'fft_score', 'sobel_score']:
                    if metric in subset.columns:
                        f.write(f"    {metric}: mean={subset[metric].mean():.2f}, "
                               f"std={subset[metric].std():.2f}\n")


# ============================================================================
# Report Generation Functions
# ============================================================================

def generate_text_report(results: Dict, output_path: str):
    """Generate comprehensive text summary report."""
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TFLITE MODEL EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for model_name, data in results.items():
            if model_name in ['endgrain', 'usability']:
                f.write(f"\n{'='*70}\n")
                f.write(f"MODEL: {model_name.upper()}\n")
                f.write(f"{'='*70}\n\n")

                metrics = data['metrics']

                f.write("DATASET:\n")
                f.write(f"  Total samples:    {metrics['total_samples']}\n")
                f.write(f"  Positive samples: {metrics['positive_samples']}\n")
                f.write(f"  Negative samples: {metrics['negative_samples']}\n")
                f.write(f"  Input shape:      {data['input_shape']}\n")
                f.write(f"  Threshold:        {data['threshold']}\n")
                f.write("\n")

                f.write("PERFORMANCE METRICS:\n")
                f.write(f"  Accuracy:         {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                f.write(f"  Precision:        {metrics['precision']:.4f}\n")
                f.write(f"  Recall:           {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score:         {metrics['f1_score']:.4f}\n")
                f.write(f"  AUC:              {metrics['auc']:.4f}\n")
                f.write("\n")

                f.write("CONFUSION MATRIX:\n")
                f.write(f"  True Negatives:   {metrics['true_negatives']}\n")
                f.write(f"  False Positives:  {metrics['false_positives']}\n")
                f.write(f"  False Negatives:  {metrics['false_negatives']}\n")
                f.write(f"  True Positives:   {metrics['true_positives']}\n")
                f.write("\n")

                f.write("ERROR RATES:\n")
                f.write(f"  False Negative Rate: {metrics['false_negative_rate']:.4f} ({metrics['false_negative_rate']*100:.2f}%)\n")
                f.write(f"  False Positive Rate: {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']*100:.2f}%)\n")
                f.write("\n")


def generate_json_summary(results: Dict, output_path: str):
    """Generate machine-readable JSON summary."""
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
                if k != 'paths'  # Exclude paths to keep file small
            }
        else:
            json_results[key] = value

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate TFLite models on test data')
    parser.add_argument('--output-dir', type=str, default='test_result',
                        help='Output directory for results')
    parser.add_argument('--models', nargs='+', default=['endgrain', 'usability'],
                        choices=['endgrain', 'usability'],
                        help='Models to evaluate')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--include-sharpness', action='store_true', default=True,
                        help='Include sharpness analysis')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TFLITE MODEL EVALUATION")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Models to evaluate: {args.models}")
    print(f"Threshold: {args.threshold}")
    print()

    # Store all results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'threshold': args.threshold
    }

    # Evaluate each model
    for model_name in args.models:
        print("\n" + "=" * 60)
        print(f"EVALUATING: {model_name.upper()}")
        print("=" * 60)

        config = MODEL_CONFIG[model_name]
        model_path = Path(config['model_path'])

        if not model_path.exists():
            # Try relative to script directory
            script_dir = Path(__file__).parent.parent
            model_path = script_dir / config['model_path']

        if not model_path.exists():
            print(f"  ERROR: Model not found at {model_path}")
            continue

        print(f"  Model path: {model_path}")

        # Run evaluation
        eval_results = batch_evaluate(
            str(model_path),
            config['test_pos'],
            config['test_neg'],
            args.threshold
        )

        # Compute metrics
        metrics = compute_metrics(
            eval_results['y_true'],
            eval_results['y_pred'],
            eval_results['y_prob']
        )

        # Print summary
        print(f"\n  Results:")
        print(f"    Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"    Precision:          {metrics['precision']:.4f}")
        print(f"    Recall:             {metrics['recall']:.4f}")
        print(f"    F1 Score:           {metrics['f1_score']:.4f}")
        print(f"    AUC:                {metrics['auc']:.4f}")
        print(f"    False Negative Rate: {metrics['false_negative_rate']:.4f}")

        # Create model output directory
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate visualizations
        print(f"\n  Generating visualizations...")

        plot_confusion_matrix(
            eval_results['y_true'],
            eval_results['y_pred'],
            config['labels'],
            str(model_output_dir / 'confusion_matrix.png'),
            model_name.title()
        )
        print(f"    - Confusion matrix saved")

        plot_roc_curve(
            eval_results['y_true'],
            eval_results['y_prob'],
            str(model_output_dir / 'roc_curve.png'),
            model_name.title()
        )
        print(f"    - ROC curve saved")

        plot_sample_predictions(
            eval_results['paths'],
            eval_results['y_true'],
            eval_results['y_pred'],
            eval_results['y_prob'],
            config['labels'],
            str(model_output_dir / 'sample_predictions.png')
        )
        print(f"    - Sample predictions saved")

        plot_false_negatives(
            eval_results['paths'],
            eval_results['y_true'],
            eval_results['y_pred'],
            eval_results['y_prob'],
            config['labels'],
            str(model_output_dir / 'false_negatives.png')
        )
        print(f"    - False negatives saved")

        # Save predictions CSV
        predictions_df = pd.DataFrame({
            'path': eval_results['paths'],
            'true_label': eval_results['y_true'],
            'predicted_label': eval_results['y_pred'],
            'probability': eval_results['y_prob']
        })
        predictions_df.to_csv(model_output_dir / 'predictions.csv', index=False)
        print(f"    - Predictions CSV saved")

        # Save metrics JSON
        with open(model_output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"    - Metrics JSON saved")

        # Store results
        all_results[model_name] = {
            'metrics': metrics,
            'input_shape': eval_results['input_shape'],
            'threshold': eval_results['threshold']
        }

    # Integrate sharpness analysis
    if args.include_sharpness:
        integrate_sharpness_analysis(PREFILTERED_PIPELINE_PATH, str(output_dir))

    # Generate summary reports
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY REPORTS")
    print("=" * 60)

    generate_text_report(all_results, str(output_dir / 'summary_report.txt'))
    print(f"  - Text report saved: {output_dir / 'summary_report.txt'}")

    generate_json_summary(all_results, str(output_dir / 'summary_metrics.json'))
    print(f"  - JSON summary saved: {output_dir / 'summary_metrics.json'}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}/")

    return all_results


if __name__ == '__main__':
    main()
