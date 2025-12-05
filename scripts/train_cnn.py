#!/usr/bin/env python3
"""
Train CNN (ResNet or EfficientNet) for wood classification

Usage:
    python scripts/train_cnn.py --model resnet18 --positive-dir dataset/wood/ --negative-dir dataset/non_wood/
    python scripts/train_cnn.py --model efficientnet_b0 --positive-dir dataset/wood/ --negative-dir dataset/non_wood/ --epochs 50

    # With augmentation stacking (2x dataset size)
    python scripts/train_cnn.py --model resnet18 --positive-dir dataset/wood/ --negative-dir dataset/non_wood/ --stack-augmentations --num-augmentations 1

    # With augmentation stacking (4x dataset size)
    python scripts/train_cnn.py --model resnet18 --positive-dir dataset/wood/ --negative-dir dataset/non_wood/ --stack-augmentations --num-augmentations 3
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_data_loaders
from src.models import ResNet18, create_efficientnet_model
from src.training.trainer import CNNTrainer


def main():
    parser = argparse.ArgumentParser(description='Train CNN for wood classification')

    # Task type
    parser.add_argument('--task', type=str, default='wood',
                       choices=['wood', 'usability'],
                       help='Task: "wood" (wood vs non-wood) or "usability" (usable vs unusable)')

    # Data arguments
    parser.add_argument('--positive-dir', type=str, required=True,
                       help='Positive class directory (wood or usable)')
    parser.add_argument('--negative-dir', type=str, required=True,
                       help='Negative class directory (non-wood or unusable)')

    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'efficientnet_b0', 'efficientnet_b1',
                               'efficientnet_b2', 'efficientnet_b3'],
                       help='Model architecture (default: resnet18)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights (default: True)')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                       help='Do not use pretrained weights')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split fraction (default: 0.2)')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience (default: 10)')

    # Data augmentation
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size (default: 224)')
    parser.add_argument('--balanced-sampler', action='store_true',
                       help='Use weighted sampler for class imbalance')
    parser.add_argument('--stack-augmentations', action='store_true',
                       help='Stack augmented images on top of originals to increase dataset size')
    parser.add_argument('--num-augmentations', type=int, default=2,
                       help='Number of augmented versions per image when stacking (default: 2)')
    parser.add_argument('--no-original', dest='include_original', action='store_false',
                       help='Do not include original images when stacking (only augmented versions)')
    parser.add_argument('--include-original', dest='include_original', action='store_true', default=True,
                       help='Include original images when stacking (default: True)')

    # Hardware
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto-detect)')

    # Output
    parser.add_argument('--output-dir', type=str, default='models/cnn',
                       help='Output directory for trained model (default: models/cnn)')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory (default: logs)')

    args = parser.parse_args()

    # Auto-set class names based on task
    if args.task == 'wood':
        args.class_names = ['non_wood', 'wood']
    else:  # usability
        args.class_names = ['unusable', 'usable']

    # Auto-generate timestamped directory name
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    auto_name = f"{args.task}_{args.model}_batch{args.batch_size}_lr{args.lr}_{timestamp}"
    args.output_dir = f"{args.output_dir}/{auto_name}"
    args.log_dir = f"{args.log_dir}/{auto_name}"

    # Setup device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print("=" * 80)
    print(f"Task: {args.task.upper()}")
    print(f"Model: {args.model.upper()}")
    print(f"Device: {device}")
    print(f"Classes: {args.class_names[0]} (0) vs {args.class_names[1]} (1)")
    print(f"Output: {args.output_dir}")
    print(f"Logs: {args.log_dir}")
    print("=" * 80)

    # Create data loaders
    print(f"\nLoading data from:")
    print(f"  Negative ({args.class_names[0]}): {args.negative_dir}")
    print(f"  Positive ({args.class_names[1]}): {args.positive_dir}")

    train_loader, val_loader = get_data_loaders(
        positive_dir=Path(args.positive_dir),
        negative_dir=Path(args.negative_dir),
        batch_size=args.batch_size,
        val_split=args.val_split,
        image_size=args.image_size,
        num_workers=args.num_workers,
        use_balanced_sampler=args.balanced_sampler,
        stack_augmentations=args.stack_augmentations,
        num_augmentations=args.num_augmentations,
        include_original=args.include_original
    )

    # Create model
    print(f"\nCreating model: {args.model}")
    if 'resnet' in args.model:
        model = ResNet18(num_classes=2, pretrained=args.pretrained)
    else:  # EfficientNet
        model = create_efficientnet_model(
            num_classes=2,
            model_name=args.model,
            pretrained=args.pretrained
        )

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Create trainer
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        save_dir=Path(args.output_dir)
    )

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)

    trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping
    )

    print("\n" + "=" * 80)
    print(f"Training complete! Model saved to {args.output_dir}/best_model.pt")
    print("=" * 80)


if __name__ == '__main__':
    main()
