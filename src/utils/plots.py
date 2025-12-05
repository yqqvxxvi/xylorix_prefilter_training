"""
Plotting utilities for training visualization
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from typing import Optional, Callable, Union
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.dataset import WoodImageDataset


def plot_training_history(history, save_path=None, show=False):
    """
    Plot training and validation loss and accuracy

    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
                 Each value is a list of metrics per epoch
        save_path: Path to save the plot (optional)
        show: Whether to display the plot (default: False)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Find best epoch (lowest validation loss)
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)
    ax1.legend(fontsize=11)

    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Mark best accuracy
    best_acc_epoch = np.argmax(history['val_acc']) + 1
    best_val_acc = max(history['val_acc'])
    ax2.axvline(x=best_acc_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_acc_epoch})')
    ax2.scatter([best_acc_epoch], [best_val_acc], color='g', s=100, zorder=5)
    ax2.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None, show=False):
    """
    Plot confusion matrix

    Args:
        cm: 2x2 confusion matrix
        class_names: List of class names ['negative', 'positive']
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix')

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(fpr, tpr, auc_score, save_path=None, show=False):
    """
    Plot ROC curve

    Args:
        fpr: False positive rate
        tpr: True positive rate
        auc_score: Area under curve score
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(feature_names, importances, top_n=20, save_path=None, show=False):
    """
    Plot feature importance for Random Forest

    Args:
        feature_names: List of feature names
        importances: Array of importance scores
        top_n: Number of top features to display
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_importances, align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_roc_curve_from_directories(
    model: Union[torch.nn.Module, object],
    positive_dir: Union[str, Path],
    negative_dir: Union[str, Path],
    transform: Optional[Callable] = None,
    batch_size: int = 32,
    device: str = 'cpu',
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    model_type: str = 'pytorch',
    class_names: list = None,
    plot_thresholds: bool = True,
    threshold_step: int = 10
) -> dict:
    """
    Create ROC curve by automatically batch processing images from two class directories

    This function loads images from positive and negative class directories,
    runs batch predictions through a model, computes ROC curve across all
    threshold values, and generates a comprehensive ROC plot.

    Args:
        model: Trained model (PyTorch or sklearn)
        positive_dir: Directory containing positive class images (label = 1)
        negative_dir: Directory containing negative class images (label = 0)
        transform: Transform to apply to images (for PyTorch models)
        batch_size: Batch size for processing (default: 32)
        device: Device to run model on ('cpu' or 'cuda')
        save_path: Path to save the ROC curve plot
        show: Whether to display the plot
        model_type: Type of model ('pytorch' or 'sklearn')
        class_names: Names of classes [negative, positive] for legend
        plot_thresholds: Whether to annotate specific threshold points
        threshold_step: Step size for threshold annotations (default: every 10th)

    Returns:
        Dictionary containing:
            - 'fpr': False positive rates
            - 'tpr': True positive rates
            - 'thresholds': Threshold values
            - 'auc': Area under curve
            - 'y_true': True labels
            - 'y_prob': Predicted probabilities

    Example:
        >>> from src.models.resnet import ResNet18
        >>> from src.data.transforms import get_inference_transforms
        >>>
        >>> # Load model
        >>> model = ResNet18(num_classes=1)
        >>> model.load_state_dict(torch.load('model.pth'))
        >>> model.eval()
        >>>
        >>> # Create ROC curve
        >>> results = create_roc_curve_from_directories(
        ...     model=model,
        ...     positive_dir='data/wood',
        ...     negative_dir='data/non_wood',
        ...     transform=get_inference_transforms(),
        ...     batch_size=32,
        ...     device='cuda',
        ...     save_path='results/roc_curve.png'
        ... )
        >>> print(f"AUC: {results['auc']:.4f}")
    """
    positive_dir = Path(positive_dir)
    negative_dir = Path(negative_dir)

    if class_names is None:
        class_names = ['Negative', 'Positive']

    print(f"Loading images from directories...")
    print(f"  Positive class: {positive_dir}")
    print(f"  Negative class: {negative_dir}")

    # Create dataset from directories
    dataset = WoodImageDataset.from_directories(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        transform=transform
    )

    # Create dataloader for batch processing
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"\nRunning batch predictions...")
    print(f"  Total images: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")

    # Collect predictions and labels
    all_labels = []
    all_probs = []

    if model_type == 'pytorch':
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(device)

                # Forward pass
                outputs = model(images)

                # Handle different output formats
                if outputs.shape[-1] == 1:
                    # Single output (binary classification with sigmoid)
                    probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                else:
                    # Multiple outputs (use softmax and take positive class)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

                all_labels.extend(labels.numpy())
                all_probs.extend(probs if isinstance(probs, list) else probs.tolist() if hasattr(probs, 'tolist') else [probs])

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} images")

    elif model_type == 'sklearn':
        # For sklearn models, extract features or use raw images
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Convert to numpy
            if isinstance(images, torch.Tensor):
                images = images.numpy()

            # Flatten images for sklearn
            images_flat = images.reshape(images.shape[0], -1)

            # Predict probabilities
            probs = model.predict_proba(images_flat)[:, 1]

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.tolist())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} images")

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'pytorch' or 'sklearn'")

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    print(f"\nComputing ROC curve...")

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"  AUC: {roc_auc:.4f}")
    print(f"  Number of threshold points: {len(thresholds)}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2.5,
            label=f'ROC curve (AUC = {roc_auc:.4f})')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier (AUC = 0.5000)')

    # Optionally plot threshold points
    if plot_thresholds and len(thresholds) > threshold_step:
        # Select threshold points to annotate
        threshold_indices = range(0, len(thresholds), threshold_step)

        for idx in threshold_indices:
            if idx < len(thresholds):
                # Skip very close points
                if idx > 0 and (fpr[idx] - fpr[idx-1] < 0.01 or tpr[idx] - tpr[idx-1] < 0.01):
                    continue

                ax.plot(fpr[idx], tpr[idx], 'ro', markersize=6, alpha=0.6)

                # Add threshold annotation
                if idx % (threshold_step * 2) == 0:  # Annotate every other point
                    ax.annotate(f'{thresholds[idx]:.2f}',
                               xy=(fpr[idx], tpr[idx]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)

    # Find optimal threshold (Youden's index)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]

    # Mark optimal point
    ax.plot(optimal_fpr, optimal_tpr, 'g*', markersize=15,
            label=f'Optimal Threshold = {optimal_threshold:.4f}\n(TPR={optimal_tpr:.3f}, FPR={optimal_fpr:.3f})')

    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title(f'ROC Curve - {class_names[1]} vs {class_names[0]}',
                 fontsize=15, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add text box with dataset info
    textstr = f'Total Images: {len(y_true)}\n'
    textstr += f'{class_names[1]}: {np.sum(y_true == 1)}\n'
    textstr += f'{class_names[0]}: {np.sum(y_true == 0)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save plot
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nROC curve saved to: {save_path}")

    # Show plot
    if show:
        plt.show()
    else:
        plt.close()

    # Return results
    results = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc,
        'y_true': y_true,
        'y_prob': y_prob,
        'optimal_threshold': optimal_threshold,
        'optimal_fpr': optimal_fpr,
        'optimal_tpr': optimal_tpr
    }

    print(f"\nResults:")
    print(f"  AUC: {roc_auc:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  TPR at optimal: {optimal_tpr:.4f}")
    print(f"  FPR at optimal: {optimal_fpr:.4f}")

    return results
