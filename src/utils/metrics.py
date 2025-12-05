"""
Evaluation metrics for classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from typing import Dict, Tuple


def compute_metrics(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Compute classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }

    # Add AUC if probabilities provided
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in formatted way

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for print statements (e.g., "Train", "Val")
    """
    prefix = f"{prefix} " if prefix else ""
    print(f"{prefix}Metrics:")
    for key, value in metrics.items():
        print(f"  {key.capitalize()}: {value:.4f}")


def get_confusion_matrix(y_true: np.ndarray,
                        y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix (2x2 for binary classification)
    """
    return confusion_matrix(y_true, y_pred)


def print_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          labels: list = ['Non-Wood', 'Wood']):
    """
    Print confusion matrix in readable format

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class names
    """
    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                {labels[0]:>10} {labels[1]:>10}")
    print(f"Actual {labels[0]:>10}  {cm[0,0]:>10} {cm[0,1]:>10}")
    print(f"       {labels[1]:>10}  {cm[1,0]:>10} {cm[1,1]:>10}")

    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()

    print(f"\nTP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Sensitivity (Recall): {tp/(tp+fn):.4f}")
    print(f"Specificity: {tn/(tn+fp):.4f}")


def print_classification_report(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                labels: list = ['Non-Wood', 'Wood']):
    """
    Print detailed classification report

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class names
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))
