"""
Train One-Class Convolutional Autoencoder for Anomaly Detection

This script trains an EfficientNet-based autoencoder on end-grain images ONLY
(unsupervised) and classifies via reconstruction error for anomaly detection.

Training Strategy:
    - Train ONLY on end-grain images (no labels)
    - Model learns to reconstruct end-grain perfectly
    - Classification via reconstruction error threshold:
        * Low error → End-grain (inlier)
        * High error → World (outlier)

Usage:
    python scripts/train_oneclass_conv_autoencoder.py \
        --endgrain_dir /path/to/endgrain \
        --world_dir /path/to/world \
        --output_dir results/conv_autoencoder \
        --epochs 100 \
        --batch_size 32
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.autoencoder.conv_model import (
    get_model,
    reconstruction_loss,
    combined_reconstruction_loss,
    vae_loss,
    PerceptualLoss
)
from src.autoencoder.vgg_model import get_vgg_model
from src.autoencoder.dataset import (
    ImageAutoencoderDataset,
    create_oneclass_dataset,
    create_anomaly_test_dataset,
    load_image_paths_from_directory
)
from src.data.transforms import get_train_transforms, get_val_transforms


class OneClassConvTrainer:
    """
    Trainer for one-class convolutional autoencoder

    Trains on end-grain images only and tracks separation between
    end-grain (low error) and world (high error) reconstruction.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        world_loader=None,
        device='cuda',
        learning_rate=0.001,
        model_type='standard',
        beta=1.0,
        perceptual_weight=0.1,
        gradient_weight=0.05,
        mse_weight=1.0,
        use_perceptual=True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.world_loader = world_loader
        self.device = device
        self.model_type = model_type
        self.beta = beta
        self.perceptual_weight = perceptual_weight
        self.gradient_weight = gradient_weight
        self.mse_weight = mse_weight
        self.use_perceptual = use_perceptual

        # Initialize perceptual loss
        self.perceptual_loss_fn = PerceptualLoss(device=device) if use_perceptual else None

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'world_loss': [],  # Optional: track world reconstruction error
            'learning_rate': [],
            'train_mse': [],
            'train_perceptual': [],
            'train_gradient': [],
            'val_mse': [],
            'val_perceptual': [],
            'val_gradient': []
        }

        if model_type == 'vae':
            self.history['train_recon_loss'] = []
            self.history['train_kl_div'] = []
            self.history['val_recon_loss'] = []
            self.history['val_kl_div'] = []

        self.best_val_loss = float('inf')

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_mse = 0
        total_perceptual = 0
        total_gradient = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch in pbar:
            images = batch.to(self.device) if not isinstance(batch, tuple) else batch[0].to(self.device)

            # Forward pass
            if self.model_type == 'vae':
                reconstructed, mu, logvar = self.model(images)
                loss, recon, kl, components = vae_loss(
                    reconstructed, images, mu, logvar, self.beta,
                    self.perceptual_loss_fn, self.perceptual_weight, self.gradient_weight, self.mse_weight
                )
                total_recon += recon.item()
                total_kl += kl.item()
                total_mse += components['mse']
                total_perceptual += components['perceptual']
                total_gradient += components['gradient']
            else:
                reconstructed, _ = self.model(images)
                loss, components = combined_reconstruction_loss(
                    reconstructed, images,
                    self.perceptual_loss_fn, self.perceptual_weight, self.gradient_weight, self.mse_weight
                )
                total_mse += components['mse']
                total_perceptual += components['perceptual']
                total_gradient += components['gradient']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / n_batches
        avg_mse = total_mse / n_batches
        avg_perceptual = total_perceptual / n_batches
        avg_gradient = total_gradient / n_batches

        if self.model_type == 'vae':
            avg_recon = total_recon / n_batches
            avg_kl = total_kl / n_batches
            return avg_loss, avg_recon, avg_kl, avg_mse, avg_perceptual, avg_gradient
        else:
            return avg_loss, avg_mse, avg_perceptual, avg_gradient

    def validate(self):
        """Validate on end-grain validation set"""
        self.model.eval()

        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_mse = 0
        total_perceptual = 0
        total_gradient = 0
        n_batches = 0
        errors = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating (End-grain)"):
                images = batch.to(self.device) if not isinstance(batch, tuple) else batch[0].to(self.device)

                # Forward pass
                if self.model_type == 'vae':
                    reconstructed, mu, logvar = self.model(images)
                    loss, recon, kl, components = vae_loss(
                        reconstructed, images, mu, logvar, self.beta,
                        self.perceptual_loss_fn, self.perceptual_weight, self.gradient_weight, self.mse_weight
                    )
                    total_recon += recon.item()
                    total_kl += kl.item()
                    total_mse += components['mse']
                    total_perceptual += components['perceptual']
                    total_gradient += components['gradient']
                else:
                    reconstructed, _ = self.model(images)
                    loss, components = combined_reconstruction_loss(
                        reconstructed, images,
                        self.perceptual_loss_fn, self.perceptual_weight, self.gradient_weight, self.mse_weight
                    )
                    total_mse += components['mse']
                    total_perceptual += components['perceptual']
                    total_gradient += components['gradient']

                total_loss += loss.item()
                n_batches += 1

                # Track individual errors
                batch_errors = torch.mean((images - reconstructed) ** 2, dim=[1, 2, 3])
                errors.extend(batch_errors.cpu().numpy().tolist())

        avg_loss = total_loss / n_batches
        avg_mse = total_mse / n_batches
        avg_perceptual = total_perceptual / n_batches
        avg_gradient = total_gradient / n_batches
        errors_array = np.array(errors)

        if self.model_type == 'vae':
            avg_recon = total_recon / n_batches
            avg_kl = total_kl / n_batches
            return avg_loss, avg_recon, avg_kl, avg_mse, avg_perceptual, avg_gradient, errors_array
        else:
            return avg_loss, avg_mse, avg_perceptual, avg_gradient, errors_array

    def validate_world(self):
        """Validate on world images (outliers) to track separation"""
        if self.world_loader is None:
            return None, None

        self.model.eval()

        total_loss = 0
        n_batches = 0
        errors = []

        with torch.no_grad():
            for batch in tqdm(self.world_loader, desc="Validating (World)"):
                images = batch.to(self.device) if not isinstance(batch, tuple) else batch[0].to(self.device)

                # Forward pass
                if self.model_type == 'vae':
                    reconstructed, mu, logvar = self.model(images)
                    loss, _, _, _ = vae_loss(
                        reconstructed, images, mu, logvar, self.beta,
                        self.perceptual_loss_fn, self.perceptual_weight, self.gradient_weight, self.mse_weight
                    )
                else:
                    reconstructed, _ = self.model(images)
                    loss, _ = combined_reconstruction_loss(
                        reconstructed, images,
                        self.perceptual_loss_fn, self.perceptual_weight, self.gradient_weight, self.mse_weight
                    )

                total_loss += loss.item()
                n_batches += 1

                # Track individual errors
                batch_errors = torch.mean((images - reconstructed) ** 2, dim=[1, 2, 3])
                errors.extend(batch_errors.cpu().numpy().tolist())

        avg_loss = total_loss / n_batches
        errors_array = np.array(errors)

        return avg_loss, errors_array

    def fit(
        self,
        epochs,
        output_dir,
        save_every=10,
        save_reconstructions=True,
        unfreeze_encoder_after=None
    ):
        """
        Train the model

        Args:
            epochs: Number of training epochs
            output_dir: Directory to save checkpoints and plots
            save_every: Save checkpoint every N epochs
            save_reconstructions: Whether to save reconstruction visualizations
            unfreeze_encoder_after: Epoch to unfreeze encoder for fine-tuning (optional)
        """
        output_dir = Path(output_dir)
        checkpoint_dir = output_dir / 'checkpoints'
        plot_dir = output_dir / 'plots'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("Training One-Class Convolutional Autoencoder")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Model type: {self.model_type}")
        print(f"Training on END-GRAIN images only (one-class)")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Loss components:")
        print(f"  - MSE: {self.mse_weight}")
        print(f"  - Perceptual: {self.perceptual_weight if self.use_perceptual else 0.0}")
        print(f"  - Gradient: {self.gradient_weight}")
        print("=" * 80)

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 80)

            # Progressive encoder unfreezing
            if unfreeze_encoder_after and epoch == unfreeze_encoder_after:
                print("\n" + "=" * 80)
                print(f"Unfreezing encoder at epoch {epoch}")
                print("=" * 80)
                self.model.unfreeze_encoder()
                # Reduce learning rate for fine-tuning
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print(f"Learning rate reduced to {self.optimizer.param_groups[0]['lr']:.6f} for fine-tuning\n")

            # Train
            if self.model_type == 'vae':
                train_loss, train_recon, train_kl, train_mse, train_perc, train_grad = self.train_epoch()
                print(f"Train Loss: {train_loss:.6f} (Recon: {train_recon:.6f}, KL: {train_kl:.6f})")
                print(f"  MSE: {train_mse:.6f}, Perceptual: {train_perc:.6f}, Gradient: {train_grad:.6f}")
                self.history['train_recon_loss'].append(train_recon)
                self.history['train_kl_div'].append(train_kl)
            else:
                train_loss, train_mse, train_perc, train_grad = self.train_epoch()
                print(f"Train Loss: {train_loss:.6f}")
                print(f"  MSE: {train_mse:.6f}, Perceptual: {train_perc:.6f}, Gradient: {train_grad:.6f}")

            self.history['train_loss'].append(train_loss)
            self.history['train_mse'].append(train_mse)
            self.history['train_perceptual'].append(train_perc)
            self.history['train_gradient'].append(train_grad)

            # Validate on end-grain
            if self.model_type == 'vae':
                val_loss, val_recon, val_kl, val_mse, val_perc, val_grad, endgrain_errors = self.validate()
                print(f"Val Loss (End-grain): {val_loss:.6f} (Recon: {val_recon:.6f}, KL: {val_kl:.6f})")
                print(f"  MSE: {val_mse:.6f}, Perceptual: {val_perc:.6f}, Gradient: {val_grad:.6f}")
                self.history['val_recon_loss'].append(val_recon)
                self.history['val_kl_div'].append(val_kl)
            else:
                val_loss, val_mse, val_perc, val_grad, endgrain_errors = self.validate()
                print(f"Val Loss (End-grain): {val_loss:.6f}")
                print(f"  MSE: {val_mse:.6f}, Perceptual: {val_perc:.6f}, Gradient: {val_grad:.6f}")

            self.history['val_loss'].append(val_loss)
            self.history['val_mse'].append(val_mse)
            self.history['val_perceptual'].append(val_perc)
            self.history['val_gradient'].append(val_grad)

            # Validate on world (optional)
            world_loss, world_errors = self.validate_world()
            if world_loss is not None:
                print(f"Val Loss (World): {world_loss:.6f}")
                self.history['world_loss'].append(world_loss)

                # Compute separation metrics
                endgrain_mean = np.mean(endgrain_errors)
                world_mean = np.mean(world_errors)
                separation = world_mean - endgrain_mean
                print(f"Reconstruction Error - End-grain: {endgrain_mean:.6f}, World: {world_mean:.6f}")
                print(f"Separation: {separation:.6f} (higher is better)")

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Step scheduler
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(checkpoint_dir / 'best_model.pth', epoch)
                print(f"✓ Saved best model (val_loss: {val_loss:.6f})")

            # Periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(checkpoint_dir / f'checkpoint_epoch_{epoch}.pth', epoch)

            # Save reconstructions
            if save_reconstructions and epoch % save_every == 0:
                self.save_reconstruction_samples(
                    plot_dir / f'reconstructions_epoch_{epoch}.png'
                )

        # Save final model
        self.save_checkpoint(checkpoint_dir / 'checkpoint_final.pth', epochs)

        # Plot training curves
        self.plot_training_curves(plot_dir / 'training_curves.png')

        # Save history
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print("\n" + "=" * 80)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        print("=" * 80)

    def save_checkpoint(self, path, epoch):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'model_type': self.model_type
        }, path)

    def save_reconstruction_samples(self, path, n_samples=8):
        """Save visualization of original vs reconstructed images"""
        self.model.eval()

        # Get a batch from validation set
        batch = next(iter(self.val_loader))
        images = batch.to(self.device) if not isinstance(batch, tuple) else batch[0].to(self.device)

        # Use actual batch size if smaller than requested n_samples
        actual_samples = min(n_samples, images.size(0))
        images = images[:actual_samples]

        # Reconstruct
        with torch.no_grad():
            if self.model_type == 'vae':
                reconstructed, _, _ = self.model(images)
            else:
                reconstructed, _ = self.model(images)

        # Create visualization
        fig, axes = plt.subplots(2, actual_samples, figsize=(actual_samples * 2, 4))

        # Handle case where actual_samples == 1
        if actual_samples == 1:
            axes = axes.reshape(2, 1)

        for i in range(actual_samples):
            # Original
            orig = images[i].cpu().permute(1, 2, 0).numpy()
            orig = np.clip(orig, 0, 1)
            axes[0, i].imshow(orig)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            # Reconstructed
            recon = reconstructed[i].cpu().permute(1, 2, 0).numpy()
            recon = np.clip(recon, 0, 1)
            axes[1, i].imshow(recon)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_training_curves(self, path):
        """Plot training curves"""
        # Create more subplots to accommodate loss component breakdown
        if self.model_type == 'vae':
            fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Flatten axes for easier indexing
        axes_flat = axes.flatten()

        # Loss curves
        ax = axes_flat[0]
        ax.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        ax.plot(self.history['val_loss'], label='Val Loss (End-grain)', linewidth=2)
        if self.history['world_loss']:
            ax.plot(self.history['world_loss'], label='Val Loss (World)', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes_flat[1]
        ax.plot(self.history['learning_rate'], linewidth=2, color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # MSE Loss components
        ax = axes_flat[2]
        ax.plot(self.history['train_mse'], label='Train MSE', linewidth=2)
        ax.plot(self.history['val_mse'], label='Val MSE', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('MSE Loss Component')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Perceptual Loss components
        ax = axes_flat[3]
        ax.plot(self.history['train_perceptual'], label='Train Perceptual', linewidth=2)
        ax.plot(self.history['val_perceptual'], label='Val Perceptual', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Perceptual Loss')
        ax.set_title('Perceptual Loss Component')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Gradient Loss components
        ax = axes_flat[4]
        ax.plot(self.history['train_gradient'], label='Train Gradient', linewidth=2)
        ax.plot(self.history['val_gradient'], label='Val Gradient', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Loss')
        ax.set_title('Gradient Loss Component')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # VAE specific plots
        if self.model_type == 'vae':
            # KL divergence
            ax = axes_flat[5]
            ax.plot(self.history['train_kl_div'], label='Train KL', linewidth=2)
            ax.plot(self.history['val_kl_div'], label='Val KL', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('KL Divergence')
            ax.set_title('VAE KL Divergence')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # Hide unused subplot for standard autoencoder
            axes_flat[5].axis('off')

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train one-class convolutional autoencoder')

    # Data arguments
    parser.add_argument('--endgrain_dir', type=str, required=True,
                       help='Directory with end-grain images (for training)')
    parser.add_argument('--world_dir', type=str, default=None,
                       help='Optional directory with world images (for threshold tuning)')
    parser.add_argument('--output_dir', type=str, default='results/conv_autoencoder',
                       help='Output directory for checkpoints and plots')

    # Model arguments
    parser.add_argument('--model_family', type=str, default='efficientnet',
                       choices=['efficientnet', 'vgg'],
                       help='Model family to use (efficientnet or vgg)')
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['standard', 'vae'],
                       help='Type of autoencoder')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                               'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                               'efficientnet_b6', 'efficientnet_b7', 'vgg16', 'vgg19'],
                       help='Model variant (EfficientNet: efficientnet_b0-b7, VGG: vgg16/vgg19)')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent space dimension')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained encoder weights')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder weights during training')
    parser.add_argument('--decoder_norm_type', type=str, default='instance',
                       choices=['none', 'batch', 'instance', 'group'],
                       help='Normalization type for decoder (VGG only, default: instance)')
    parser.add_argument('--unfreeze_encoder_after', type=int, default=None,
                       help='Epoch to unfreeze encoder for fine-tuning (optional)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta weight for VAE KL divergence')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')

    # Loss function arguments
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                       help='Weight for perceptual loss component')
    parser.add_argument('--gradient_weight', type=float, default=0.05,
                       help='Weight for gradient loss component')
    parser.add_argument('--mse_weight', type=float, default=1.0,
                       help='Weight for MSE loss component (set to 0.0 for perceptual-only)')
    parser.add_argument('--disable_perceptual', action='store_true',
                       help='Disable perceptual loss (uses only MSE and gradient loss)')

    # Data loading arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--cache_images', action='store_true',
                       help='Cache images in memory (faster but uses RAM)')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("Configuration")
    print("=" * 80)
    for key, value in vars(args).items():
        print(f"{key:20s}: {value}")
    print("=" * 80)

    # Get transforms
    train_transform = get_train_transforms(image_size=224, grayscale=False)
    val_transform = get_val_transforms(image_size=224, grayscale=False)

    # Create datasets (ONE-CLASS: end-grain only)
    print("\n" + "=" * 80)
    print("Loading Datasets")
    print("=" * 80)
    print("Training on END-GRAIN images only (one-class anomaly detection)")

    train_dataset, val_dataset = create_oneclass_dataset(
        args.endgrain_dir,
        transform=train_transform,
        val_split=args.val_split,
        seed=args.seed,
        cache_images=args.cache_images
    )

    # Optional: Load world images for threshold tuning
    world_dataset = None
    if args.world_dir is not None:
        print(f"\nLoading WORLD images from {args.world_dir} for threshold tuning")
        world_paths = load_image_paths_from_directory(args.world_dir)
        print(f"Found {len(world_paths)} world images")

        world_dataset = ImageAutoencoderDataset(
            world_paths,
            transform=val_transform,
            labels=None,
            cache_images=args.cache_images
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    world_loader = None
    if world_dataset is not None:
        world_loader = DataLoader(
            world_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # Create model
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    if args.model_family == 'efficientnet':
        model = get_model(
            model_type=args.model_type,
            latent_dim=args.latent_dim,
            model_name=args.model_name,
            pretrained=args.pretrained,
            freeze_encoder=args.freeze_encoder
        )
    elif args.model_family == 'vgg':
        model = get_vgg_model(
            model_type=args.model_type,
            latent_dim=args.latent_dim,
            vgg_variant=args.model_name,
            pretrained=args.pretrained,
            freeze_encoder=args.freeze_encoder,
            decoder_norm_type=args.decoder_norm_type
        )
    else:
        raise ValueError(f"Unknown model_family: {args.model_family}. Choose 'efficientnet' or 'vgg'")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = OneClassConvTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        world_loader=world_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        beta=args.beta,
        perceptual_weight=args.perceptual_weight,
        gradient_weight=args.gradient_weight,
        mse_weight=args.mse_weight,
        use_perceptual=not args.disable_perceptual
    )

    # Train
    trainer.fit(
        epochs=args.epochs,
        output_dir=output_dir,
        save_every=args.save_every,
        unfreeze_encoder_after=args.unfreeze_encoder_after
    )


if __name__ == '__main__':
    main()
