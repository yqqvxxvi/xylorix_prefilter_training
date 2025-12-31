"""
Train contrastive learning model for wood classification

Self-supervised pre-training using SimCLR on wood images.
After pre-training, the encoder can be used for downstream tasks.
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import WoodImageDataset
from src.contrastive import (
    get_contrastive_augmentation,
    ContrastiveTransformations,
    ContrastiveDataset,
    contrastive_collate_fn,
    NTXentLoss,
    SimCLRModel
)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch

    Args:
        model: SimCLR model
        dataloader: Training dataloader
        criterion: NT-Xent loss
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, views in enumerate(pbar):
        # views is a list of [view1_batch, view2_batch]
        # Each view has shape (batch_size, channels, height, width)
        view1, view2 = views[0].to(device), views[1].to(device)

        # Forward pass through encoder + projection head
        # We use the projection outputs (z1, z2) for contrastive loss
        _, z1 = model(view1)
        _, z2 = model(view2)

        # Compute NT-Xent loss
        # z1 and z2 are the positive pairs (augmentations of the same image)
        loss = criterion(z1, z2)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train SimCLR for wood classification')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing positive/ and negative/ subdirectories')
    parser.add_argument('--positive_dir', type=str, default='positive',
                        help='Name of positive class directory (default: positive)')
    parser.add_argument('--negative_dir', type=str, default='negative',
                        help='Name of negative class directory (default: negative)')

    # Model arguments
    parser.add_argument('--encoder', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50', 'resnet101', 'efficientnet_b0'],
                        help='Encoder architecture (default: resnet50)')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Projection head output dimension (default: 128)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Use grayscale images (1 channel)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64, use larger if possible)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate (default: 0.0003)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for NT-Xent loss (default: 0.5)')
    parser.add_argument('--augmentation_strength', type=str, default='strong',
                        choices=['weak', 'medium', 'strong'],
                        help='Augmentation strength (default: strong)')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--output_dir', type=str, default='outputs/contrastive',
                        help='Output directory for checkpoints (default: outputs/contrastive)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("SimCLR Contrastive Learning for Wood Classification")
    print("=" * 60)
    print(f"Encoder: {args.encoder}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Temperature: {args.temperature}")
    print(f"Augmentation strength: {args.augmentation_strength}")
    print(f"Grayscale: {args.grayscale}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Setup device
    device = torch.device(args.device)

    # Create augmentation pipeline
    base_transform = get_contrastive_augmentation(
        image_size=224,
        grayscale=args.grayscale,
        strength=args.augmentation_strength
    )
    contrastive_transform = ContrastiveTransformations(base_transform, n_views=2)

    # Load dataset
    print("\nLoading dataset...")
    data_root = Path(args.data_dir)
    positive_dir = data_root / args.positive_dir
    negative_dir = data_root / args.negative_dir

    # Create base dataset (without transforms)
    base_dataset = WoodImageDataset.from_directories(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        transform=None,  # No transform here, will apply in ContrastiveDataset
        grayscale=args.grayscale
    )

    # Wrap with contrastive dataset
    train_dataset = ContrastiveDataset(
        base_dataset=base_dataset,
        transform=contrastive_transform,
        n_views=2,
        return_label=False  # Don't need labels for self-supervised learning
    )

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=contrastive_collate_fn,
        pin_memory=True if args.device == 'cuda' else False,
        drop_last=True  # Drop last incomplete batch for stable training
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Create model
    print("\nInitializing model...")
    model = SimCLRModel(
        encoder_name=args.encoder,
        pretrained=False,  # Self-supervised, no ImageNet pretraining
        projection_dim=args.projection_dim,
        grayscale=args.grayscale
    ).to(device)

    print(f"Encoder output dim: {model.encoder_dim}")
    print(f"Projection output dim: {model.projection_dim}")

    # Create loss and optimizer
    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Optional: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    train_losses = []
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train one epoch
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(avg_loss)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # Save checkpoint
        if epoch % args.save_freq == 0 or avg_loss < best_loss:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = output_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss,
                    'args': vars(args)
                }, best_path)
                print(f"New best model saved! (loss: {avg_loss:.4f})")

    # Save final model
    final_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'loss': train_losses[-1],
        'args': vars(args)
    }, final_path)
    print(f"\nFinal model saved to {final_path}")

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('NT-Xent Loss', fontsize=12)
    plt.title('SimCLR Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / 'training_curve.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Training curve saved to {plot_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Evaluate the learned representations using linear probe")
    print("2. Fine-tune the encoder on downstream classification task")
    print("3. Use the encoder for feature extraction")


if __name__ == '__main__':
    main()
