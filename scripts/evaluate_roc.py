#!/usr/bin/env python3
"""
Example script to generate ROC curves from two class directories

This script demonstrates how to use the create_roc_curve_from_directories
function to evaluate model performance across different threshold values.
"""

import sys
from pathlib import Path
import torch
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.resnet import ResNet18
from src.models.efficientnet import EfficientNetClassifier
from src.data.transforms import get_inference_transforms
from src.utils.plots import create_roc_curve_from_directories


def main():
    parser = argparse.ArgumentParser(description='Generate ROC curve from two class directories')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--positive_dir', type=str, required=True,
                        help='Directory containing positive class images')
    parser.add_argument('--negative_dir', type=str, required=True,
                        help='Directory containing negative class images')

    # Optional arguments
    parser.add_argument('--model_type', type=str, default='resnet18',
                        choices=['resnet18', 'efficientnet'],
                        help='Model architecture')
    parser.add_argument('--output_path', type=str, default='results/roc_curve.png',
                        help='Path to save ROC curve plot')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--class_names', type=str, nargs=2, default=['Unusable', 'Usable'],
                        help='Class names [negative positive]')
    parser.add_argument('--plot_thresholds', action='store_true',
                        help='Annotate threshold values on plot')
    parser.add_argument('--threshold_step', type=int, default=10,
                        help='Step size for threshold annotations')
    parser.add_argument('--show', action='store_true',
                        help='Display the plot after generation')

    args = parser.parse_args()

    print("=" * 70)
    print("ROC Curve Generation")
    print("=" * 70)

    # Load checkpoint first to determine num_classes
    print(f"\nLoading checkpoint: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=args.device)

    # Detect number of classes from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Find the final layer to determine num_classes
    num_classes = None
    if args.model_type == 'resnet18':
        if 'fc.weight' in state_dict:
            num_classes = state_dict['fc.weight'].shape[0]
    elif args.model_type == 'efficientnet':
        if 'model.classifier.1.weight' in state_dict:
            num_classes = state_dict['model.classifier.1.weight'].shape[0]

    if num_classes is None:
        print("  Warning: Could not auto-detect num_classes from checkpoint")
        print("  Trying num_classes=2 (standard binary classification)")
        num_classes = 2

    print(f"  Detected num_classes: {num_classes}")
    if 'epoch' in checkpoint:
        print(f"  Checkpoint from epoch: {checkpoint['epoch']}")

    # Create model with correct num_classes
    print(f"\nCreating model: {args.model_type}")
    if args.model_type == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif args.model_type == 'efficientnet':
        model = EfficientNetClassifier(num_classes=num_classes, model_name='efficientnet_b0', pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Load checkpoint weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Model loaded successfully")

    # Get transforms
    transform = get_inference_transforms(image_size=args.image_size)

    # Generate ROC curve
    print(f"\nGenerating ROC curve...")
    results = create_roc_curve_from_directories(
        model=model,
        positive_dir=args.positive_dir,
        negative_dir=args.negative_dir,
        transform=transform,
        batch_size=args.batch_size,
        device=args.device,
        save_path=args.output_path,
        show=args.show,
        model_type='pytorch',
        class_names=args.class_names,
        plot_thresholds=args.plot_thresholds,
        threshold_step=args.threshold_step
    )

    print("\n" + "=" * 70)
    print("ROC Curve Generation Complete")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  AUC Score: {results['auc']:.4f}")
    print(f"  Optimal Threshold: {results['optimal_threshold']:.4f}")
    print(f"  True Positive Rate @ Optimal: {results['optimal_tpr']:.4f}")
    print(f"  False Positive Rate @ Optimal: {results['optimal_fpr']:.4f}")
    print(f"  Plot saved to: {args.output_path}")

    # Print some threshold examples
    print(f"\nSample Thresholds:")
    num_samples = min(5, len(results['thresholds']))
    indices = [int(i * len(results['thresholds']) / num_samples) for i in range(num_samples)]
    for idx in indices:
        if idx < len(results['thresholds']):
            print(f"  Threshold: {results['thresholds'][idx]:.4f} -> "
                  f"TPR: {results['tpr'][idx]:.4f}, FPR: {results['fpr'][idx]:.4f}")


if __name__ == '__main__':
    main()
