"""
Training script for end-grain feature autoencoder

This script trains an autoencoder on extracted features from end-grain
and world images, learning a compact representation for efficient
end-grain detection.

Usage:
    python scripts/train_autoencoder.py --endgrain_dir path/to/endgrain \
                                        --world_dir path/to/world \
                                        --output_dir results/autoencoder \
                                        --epochs 100
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.autoencoder import EndGrainFeatureExtractor
from src.autoencoder.model import FeatureAutoencoder, VariationalAutoencoder, vae_loss
from src.autoencoder.dataset import (
    FeatureDataset,
    create_data_loaders,
    load_image_paths_from_directory
)


class AutoencoderTrainer:
    """
    Trainer class for autoencoder models

    Handles training loop, validation, checkpointing, and visualization.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        output_dir: Path = Path('results/autoencoder'),
        model_type: str = 'standard'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.model_type = model_type

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        if model_type == 'variational':
            self.history['train_recon_loss'] = []
            self.history['train_kl_loss'] = []
            self.history['val_recon_loss'] = []
            self.history['val_kl_loss'] = []

        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')

        for batch in pbar:
            # Handle both labeled and unlabeled data
            if isinstance(batch, tuple):
                features, _ = batch
            else:
                features = batch

            features = features.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.model_type == 'variational':
                reconstructed, mu, logvar = self.model(features)
                loss, recon_loss, kl_loss = vae_loss(reconstructed, features, mu, logvar, beta=1.0)
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
            else:
                reconstructed, _ = self.model(features)
                loss = self.criterion(reconstructed, features)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        # Compute average losses
        avg_loss = total_loss / len(self.train_loader)
        result = {'loss': avg_loss}

        if self.model_type == 'variational':
            result['recon_loss'] = total_recon_loss / len(self.train_loader)
            result['kl_loss'] = total_kl_loss / len(self.train_loader)

        return result

    def validate(self, epoch: int) -> dict:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')

            for batch in pbar:
                # Handle both labeled and unlabeled data
                if isinstance(batch, tuple):
                    features, _ = batch
                else:
                    features = batch

                features = features.to(self.device)

                # Forward pass
                if self.model_type == 'variational':
                    reconstructed, mu, logvar = self.model(features)
                    loss, recon_loss, kl_loss = vae_loss(reconstructed, features, mu, logvar, beta=1.0)
                    total_recon_loss += recon_loss.item()
                    total_kl_loss += kl_loss.item()
                else:
                    reconstructed, _ = self.model(features)
                    loss = self.criterion(reconstructed, features)

                total_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        # Compute average losses
        avg_loss = total_loss / len(self.val_loader)
        result = {'loss': avg_loss}

        if self.model_type == 'variational':
            result['recon_loss'] = total_recon_loss / len(self.val_loader)
            result['kl_loss'] = total_kl_loss / len(self.val_loader)

        return result

    def train(self, num_epochs: int):
        """Main training loop"""
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model type: {self.model_type}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(current_lr)

            if self.model_type == 'variational':
                self.history['train_recon_loss'].append(train_metrics['recon_loss'])
                self.history['train_kl_loss'].append(train_metrics['kl_loss'])
                self.history['val_recon_loss'].append(val_metrics['recon_loss'])
                self.history['val_kl_loss'].append(val_metrics['kl_loss'])

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Val Loss:   {val_metrics['loss']:.6f}")
            print(f"  LR:         {current_lr:.6f}")

            if self.model_type == 'variational':
                print(f"  Train Recon: {train_metrics['recon_loss']:.6f}, KL: {train_metrics['kl_loss']:.6f}")
                print(f"  Val Recon:   {val_metrics['recon_loss']:.6f}, KL: {val_metrics['kl_loss']:.6f}")

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model saved (val_loss: {self.best_val_loss:.6f})")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Plot training curves every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_training_curves()

        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        # Final save
        self.save_checkpoint(num_epochs - 1, is_best=False, suffix='final')
        self.plot_training_curves()
        self.save_history()

    def save_checkpoint(self, epoch: int, is_best: bool = False, suffix: str = ''):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }

        if is_best:
            path = self.output_dir / 'checkpoints' / 'best_model.pth'
        elif suffix:
            path = self.output_dir / 'checkpoints' / f'checkpoint_{suffix}.pth'
        else:
            path = self.output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth'

        torch.save(checkpoint, path)

    def save_history(self):
        """Save training history to JSON"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_training_curves(self):
        """Plot and save training curves"""
        if self.model_type == 'variational':
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(self.history['train_loss']) + 1)

        if self.model_type == 'standard':
            # Loss curve
            axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', marker='o', markersize=3)
            axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', marker='s', markersize=3)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss (MSE)')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Learning rate
            axes[1].plot(epochs, self.history['learning_rate'], color='green', marker='o', markersize=3)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)

        else:  # VAE
            axes = axes.flatten()

            # Total loss
            axes[0].plot(epochs, self.history['train_loss'], label='Train', marker='o', markersize=3)
            axes[0].plot(epochs, self.history['val_loss'], label='Val', marker='s', markersize=3)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Total Loss')
            axes[0].set_title('Total Loss (Recon + KL)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Reconstruction loss
            axes[1].plot(epochs, self.history['train_recon_loss'], label='Train', marker='o', markersize=3)
            axes[1].plot(epochs, self.history['val_recon_loss'], label='Val', marker='s', markersize=3)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Reconstruction Loss')
            axes[1].set_title('Reconstruction Loss (MSE)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # KL divergence
            axes[2].plot(epochs, self.history['train_kl_loss'], label='Train', marker='o', markersize=3)
            axes[2].plot(epochs, self.history['val_kl_loss'], label='Val', marker='s', markersize=3)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('KL Divergence')
            axes[2].set_title('KL Divergence')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            # Learning rate
            axes[3].plot(epochs, self.history['learning_rate'], color='green', marker='o', markersize=3)
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Learning Rate')
            axes[3].set_title('Learning Rate Schedule')
            axes[3].set_yscale('log')
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train autoencoder on end-grain features')

    # Data arguments
    parser.add_argument('--endgrain_dir', type=str, required=True,
                       help='Directory containing end-grain images')
    parser.add_argument('--world_dir', type=str, required=True,
                       help='Directory containing world (non-wood) images')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['standard', 'variational'],
                       help='Autoencoder type (default: standard)')
    parser.add_argument('--latent_dim', type=int, default=16,
                       help='Latent space dimension (default: 16)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32],
                       help='Hidden layer dimensions (default: 128 64 32)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers (default: 4)')

    # Feature extraction arguments
    parser.add_argument('--normalize_features', action='store_true',
                       help='Normalize features (z-score)')
    parser.add_argument('--cache_features', action='store_true',
                       help='Cache extracted features in memory')
    parser.add_argument('--precompute', action='store_true',
                       help='Precompute all features before training')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results/autoencoder',
                       help='Output directory (default: results/autoencoder)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = 'cpu'

    print("=" * 80)
    print("End-Grain Feature Autoencoder Training")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)

    # Initialize feature extractor
    print("\nInitializing feature extractor...")
    extractor = EndGrainFeatureExtractor(
        image_size=224,
        normalize=args.normalize_features
    )
    feature_dim = extractor.get_feature_count()
    print(f"Feature dimension: {feature_dim}")

    # Load image paths
    print("\nLoading image paths...")
    endgrain_paths = load_image_paths_from_directory(args.endgrain_dir)
    world_paths = load_image_paths_from_directory(args.world_dir)
    print(f"Found {len(endgrain_paths)} end-grain images")
    print(f"Found {len(world_paths)} world images")

    # Create datasets
    print("\nCreating datasets...")
    all_paths = endgrain_paths + world_paths

    # Shuffle
    np.random.seed(args.seed)
    indices = np.random.permutation(len(all_paths))
    all_paths = [all_paths[i] for i in indices]

    # Split
    n_val = int(len(all_paths) * args.val_split)
    train_paths = all_paths[n_val:]
    val_paths = all_paths[:n_val]

    print(f"Train: {len(train_paths)} images")
    print(f"Val: {len(val_paths)} images")

    train_dataset = FeatureDataset(
        train_paths,
        extractor,
        cache_features=args.cache_features
    )

    val_dataset = FeatureDataset(
        val_paths,
        extractor,
        cache_features=args.cache_features
    )

    # Precompute features if requested
    if args.precompute:
        print("\nPrecomputing features...")
        train_dataset.precompute_features(verbose=True)
        val_dataset.precompute_features(verbose=True)

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create model
    print("\nInitializing model...")
    if args.model_type == 'standard':
        model = FeatureAutoencoder(
            input_dim=feature_dim,
            latent_dim=args.latent_dim,
            hidden_dims=args.hidden_dims
        )
    else:
        model = VariationalAutoencoder(
            input_dim=feature_dim,
            latent_dim=args.latent_dim,
            hidden_dims=args.hidden_dims
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = AutoencoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        model_type=args.model_type
    )

    # Save configuration
    config_path = Path(args.output_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Train
    trainer.train(num_epochs=args.epochs)

    print("\nTraining complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
