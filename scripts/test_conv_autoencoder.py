"""
Comprehensive Test and Visualization for Convolutional Autoencoder

This script provides 4 comprehensive visualizations for testing one-class
anomaly detection models:

1. Reconstruction error distribution (end-grain vs world)
2. Sample reconstructions with pixel-wise difference heatmaps
3. Confusion matrix and classification metrics
4. ROC curve and threshold analysis

Usage:
    python scripts/test_conv_autoencoder.py \
        --checkpoint results/conv_autoencoder/checkpoints/best_model.pth \
        --endgrain_test_dir /path/to/endgrain/test \
        --world_test_dir /path/to/world/test \
        --output_dir results/test_viz
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve
)
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.autoencoder.conv_model import (
    EfficientNetAutoencoder,
    EfficientNetVAE
)
from src.autoencoder.dataset import (
    create_anomaly_test_dataset
)
from src.data.transforms import get_val_transforms


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Load model from checkpoint

    Returns:
        model, model_type, checkpoint_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Determine model type
    model_type = checkpoint.get('model_type', 'standard')

    # Get architecture parameters from state dict
    state_dict = checkpoint['model_state_dict']

    # Determine if VAE
    has_fc_mu = any('fc_mu' in key for key in state_dict.keys())
    if has_fc_mu:
        model_type = 'vae'

    # Get latent dim
    if has_fc_mu:
        latent_dim = state_dict['fc_mu.weight'].shape[0]
    else:
        # Find from fc_encode
        latent_dim = state_dict['fc_encode.weight'].shape[0]

    # Get encoder dim (determine model variant)
    encoder_dim = state_dict['fc_decode.weight'].shape[1] if 'fc_decode.weight' in state_dict else 1280

    # Map encoder_dim to model_name
    dim_to_model = {
        1280: 'efficientnet_b0',
        1408: 'efficientnet_b2',
        1536: 'efficientnet_b3',
        1792: 'efficientnet_b4',
        2048: 'efficientnet_b5',
        2304: 'efficientnet_b6',
        2560: 'efficientnet_b7'
    }
    model_name = dim_to_model.get(encoder_dim, 'efficientnet_b0')

    # Create model
    if model_type == 'vae':
        model = EfficientNetVAE(
            latent_dim=latent_dim,
            model_name=model_name,
            pretrained=False
        )
    else:
        model = EfficientNetAutoencoder(
            latent_dim=latent_dim,
            model_name=model_name,
            pretrained=False
        )

    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_type} model")
    print(f"Architecture: {model_name}")
    print(f"Latent dim: {latent_dim}")

    return model, model_type, checkpoint


def compute_reconstruction_errors(model, dataloader, model_type, device):
    """
    Compute reconstruction error for all samples

    Returns:
        errors: numpy array of reconstruction errors (MSE per image)
        labels: numpy array of labels if available
    """
    model.eval()
    errors = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing reconstruction errors"):
            # Handle different batch formats (tuple, list, or tensor)
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, batch_labels = batch[0], batch[1]
                images = images.to(device)
                # Handle labels (could be tensor or list)
                if torch.is_tensor(batch_labels):
                    labels_list.extend(batch_labels.cpu().numpy())
                else:
                    labels_list.extend(batch_labels)
            else:
                # Single tensor (no labels)
                images = batch.to(device) if torch.is_tensor(batch) else batch[0].to(device)

            # Reconstruct
            if model_type == 'vae':
                reconstructed, _, _ = model(images)
            else:
                reconstructed, _ = model(images)

            # Compute per-image MSE
            batch_errors = torch.mean((images - reconstructed) ** 2, dim=[1, 2, 3])
            errors.extend(batch_errors.cpu().numpy())

    errors = np.array(errors)
    labels = np.array(labels_list) if labels_list else None

    return errors, labels


def find_optimal_threshold(errors, labels):
    """
    Find optimal threshold that maximizes F1 score

    Returns:
        optimal_threshold, f1_score
    """
    fpr, tpr, thresholds = roc_curve(labels, -errors)  # Negative because low error = positive class

    # Compute F1 for each threshold
    best_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        predictions = (errors < threshold).astype(int)
        f1 = f1_score(labels, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================================
# Visualization 1: Reconstruction Error Distribution
# ============================================================================

def plot_error_distribution(
    endgrain_errors,
    world_errors,
    threshold,
    output_path
):
    """
    Plot histogram of reconstruction errors for both classes

    Shows clear separation between end-grain (low error) and world (high error)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms
    bins = np.linspace(
        min(endgrain_errors.min(), world_errors.min()),
        max(endgrain_errors.max(), world_errors.max()),
        50
    )

    ax.hist(endgrain_errors, bins=bins, alpha=0.6, label='End-grain (Inliers)',
            color='#4ECDC4', edgecolor='black')
    ax.hist(world_errors, bins=bins, alpha=0.6, label='World (Outliers)',
            color='#FF6B6B', edgecolor='black')

    # Add threshold line
    ax.axvline(threshold, color='green', linestyle='--', linewidth=2,
               label=f'Threshold: {threshold:.6f}')

    # Statistics
    endgrain_mean = np.mean(endgrain_errors)
    world_mean = np.mean(world_errors)
    separation = world_mean - endgrain_mean

    ax.axvline(endgrain_mean, color='#4ECDC4', linestyle=':', linewidth=1.5,
               alpha=0.7, label=f'End-grain mean: {endgrain_mean:.6f}')
    ax.axvline(world_mean, color='#FF6B6B', linestyle=':', linewidth=1.5,
               alpha=0.7, label=f'World mean: {world_mean:.6f}')

    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Reconstruction Error Distribution\n'
                 f'Separation: {separation:.6f} (higher is better)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved error distribution plot to: {output_path}")


# ============================================================================
# Visualization 2: Sample Reconstructions
# ============================================================================

def plot_sample_reconstructions(
    model,
    dataset,
    model_type,
    device,
    output_path,
    n_samples=8
):
    """
    Plot original vs reconstructed images with difference heatmaps
    """
    model.eval()

    # Sample images (4 endgrain + 4 world)
    endgrain_indices = [i for i, label in enumerate(dataset.labels) if label == 1][:4]
    world_indices = [i for i, label in enumerate(dataset.labels) if label == 0][:4]
    sample_indices = endgrain_indices + world_indices

    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2, 6))

    with torch.no_grad():
        for col, idx in enumerate(sample_indices):
            image, label = dataset[idx]
            image = image.unsqueeze(0).to(device)

            # Reconstruct
            if model_type == 'vae':
                reconstructed, _, _ = model(image)
            else:
                reconstructed, _ = model(image)

            # Convert to numpy
            orig = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
            recon = reconstructed.cpu().squeeze(0).permute(1, 2, 0).numpy()

            # Clip to valid range
            orig = np.clip(orig, 0, 1)
            recon = np.clip(recon, 0, 1)

            # Compute difference
            diff = np.abs(orig - recon)
            error = np.mean((orig - recon) ** 2)

            # Plot original
            axes[0, col].imshow(orig)
            axes[0, col].axis('off')
            if col == 0:
                axes[0, col].set_ylabel('Original', fontsize=10, fontweight='bold')
            label_text = "End-grain" if label == 1 else "World"
            axes[0, col].set_title(f'{label_text}\nError: {error:.6f}', fontsize=9)

            # Plot reconstructed
            axes[1, col].imshow(recon)
            axes[1, col].axis('off')
            if col == 0:
                axes[1, col].set_ylabel('Reconstructed', fontsize=10, fontweight='bold')

            # Plot difference heatmap
            diff_gray = np.mean(diff, axis=2)
            im = axes[2, col].imshow(diff_gray, cmap='hot', vmin=0, vmax=0.5)
            axes[2, col].axis('off')
            if col == 0:
                axes[2, col].set_ylabel('Difference', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved sample reconstructions to: {output_path}")


# ============================================================================
# Visualization 3: Confusion Matrix & Metrics
# ============================================================================

def plot_confusion_matrix_and_metrics(
    errors,
    labels,
    threshold,
    output_path
):
    """
    Plot confusion matrix and classification metrics
    """
    # Make predictions based on threshold
    predictions = (errors < threshold).astype(int)

    # Compute metrics
    cm = confusion_matrix(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot confusion matrix
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['World', 'End-grain'],
                yticklabels=['World', 'End-grain'],
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Plot metrics
    ax = axes[1]
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    bars = ax.bar(metrics.keys(), metrics.values(), color=['#4ECDC4', '#FF6B6B', '#FFD93D', '#95E1D3'])
    ax.set_ylim([0, 1.0])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved confusion matrix and metrics to: {output_path}")
    print(f"\nClassification Metrics (threshold={threshold:.6f}):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")


# ============================================================================
# Visualization 4: ROC Curve & Threshold Analysis
# ============================================================================

def plot_roc_and_threshold_analysis(
    errors,
    labels,
    optimal_threshold,
    output_path
):
    """
    Plot ROC curve and precision-recall curve with threshold analysis
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    ax = axes[0]
    fpr, tpr, thresholds = roc_curve(labels, -errors)  # Negative because low error = positive
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    # Mark optimal threshold point
    optimal_idx = np.argmin(np.abs(thresholds + optimal_threshold))  # Negative errors
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
            label=f'Optimal Threshold: {optimal_threshold:.6f}')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Precision-Recall Curve
    ax = axes[1]
    precision, recall, pr_thresholds = precision_recall_curve(labels, -errors)
    pr_auc = auc(recall, precision)

    ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved ROC curve and PR curve to: {output_path}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test and visualize convolutional autoencoder')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--endgrain_test_dir', type=str, required=True,
                       help='Directory with end-grain test images')
    parser.add_argument('--world_test_dir', type=str, required=True,
                       help='Directory with world test images')
    parser.add_argument('--output_dir', type=str, default='results/test_viz',
                       help='Output directory for visualizations')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Optional manual threshold (if not provided, will be computed)')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = 'cpu'

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Convolutional Autoencoder Testing and Visualization")
    print("=" * 80)

    # Load model
    print("\nLoading model from checkpoint...")
    model, model_type, checkpoint = load_model_from_checkpoint(
        args.checkpoint,
        device=args.device
    )

    # Get transforms
    transform = get_val_transforms(image_size=224, grayscale=False)

    # Create test dataset with labels
    print("\nLoading test dataset...")
    test_dataset = create_anomaly_test_dataset(
        args.endgrain_test_dir,
        args.world_test_dir,
        transform=transform,
        cache_images=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Compute reconstruction errors
    print("\nComputing reconstruction errors...")
    errors, labels = compute_reconstruction_errors(
        model,
        test_loader,
        model_type,
        args.device
    )

    # Separate errors by class
    endgrain_errors = errors[labels == 1]
    world_errors = errors[labels == 0]

    print(f"\nEnd-grain errors: mean={np.mean(endgrain_errors):.6f}, std={np.std(endgrain_errors):.6f}")
    print(f"World errors: mean={np.mean(world_errors):.6f}, std={np.std(world_errors):.6f}")
    print(f"Separation: {np.mean(world_errors) - np.mean(endgrain_errors):.6f}")

    # Find optimal threshold if not provided
    if args.threshold is None:
        print("\nFinding optimal threshold...")
        threshold, best_f1 = find_optimal_threshold(errors, labels)
        print(f"Optimal threshold: {threshold:.6f} (F1={best_f1:.4f})")
    else:
        threshold = args.threshold
        print(f"\nUsing manual threshold: {threshold:.6f}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    # Visualization 1: Error distribution
    print("\n1. Reconstruction error distribution...")
    plot_error_distribution(
        endgrain_errors,
        world_errors,
        threshold,
        output_dir / 'error_distribution.png'
    )

    # Visualization 2: Sample reconstructions
    print("\n2. Sample reconstructions...")
    plot_sample_reconstructions(
        model,
        test_dataset,
        model_type,
        args.device,
        output_dir / 'sample_reconstructions.png'
    )

    # Visualization 3: Confusion matrix & metrics
    print("\n3. Confusion matrix and metrics...")
    plot_confusion_matrix_and_metrics(
        errors,
        labels,
        threshold,
        output_dir / 'confusion_matrix.png'
    )

    # Visualization 4: ROC curve & threshold analysis
    print("\n4. ROC curve and threshold analysis...")
    plot_roc_and_threshold_analysis(
        errors,
        labels,
        threshold,
        output_dir / 'roc_curve.png'
    )

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - error_distribution.png")
    print("  - sample_reconstructions.png")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("=" * 80)


if __name__ == '__main__':
    main()
