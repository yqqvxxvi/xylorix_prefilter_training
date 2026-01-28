"""
Deep Convolutional Autoencoder for One-Class Anomaly Detection

This module implements EfficientNet-based autoencoders for end-grain wood detection
using one-class classification via reconstruction error.

Architecture:
    - Encoder: Pretrained EfficientNet-B0 (from ImageNet)
    - Decoder: Custom transposed convolutions for image reconstruction
    - Training: Unsupervised on end-grain images only
    - Inference: Low reconstruction error = end-grain, High error = world (outlier)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights
from typing import Tuple, Optional
import warnings


class ConvTransposeBlock(nn.Module):
    """
    Transposed convolution block for decoder upsampling

    Architecture: ConvTranspose2d → BatchNorm2d → ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        use_batchnorm: bool = True,
        activation: bool = True
    ):
        super(ConvTransposeBlock, self).__init__()

        layers = []

        # Transposed convolution
        layers.append(nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=not use_batchnorm
        ))

        # Batch normalization
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        # Activation
        if activation:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features

    Compares high-level features between reconstructed and original images
    to improve perceptual quality beyond pixel-wise MSE.

    Args:
        device: Device to run the VGG16 model on

    Example:
        >>> perceptual_loss_fn = PerceptualLoss(device='cuda')
        >>> loss = perceptual_loss_fn(reconstructed, original)
    """

    def __init__(self, device='cuda'):
        super().__init__()
        # Load pretrained VGG16 and extract features up to layer 16
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(device)
        self.vgg = vgg.eval()

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss

        Args:
            pred: Predicted/reconstructed images (batch, 3, H, W)
            target: Target/original images (batch, 3, H, W)

        Returns:
            Perceptual loss value
        """
        # Extract features
        pred_feat = self.vgg(pred)
        target_feat = self.vgg(target)

        # Compute MSE on features
        return F.mse_loss(pred_feat, target_feat)


class EfficientNetAutoencoder(nn.Module):
    """
    EfficientNet-based autoencoder for one-class anomaly detection

    Trains on end-grain images only and classifies via reconstruction error:
    - Low reconstruction error → End-grain (inlier)
    - High reconstruction error → World/Non-wood (outlier)

    Architecture:
        Encoder: EfficientNet-B0 (pretrained) → 1280 features
        Bottleneck: AdaptiveAvgPool → Linear(1280 → latent_dim)
        Decoder: Linear(latent_dim → 1280×7×7) → 5 ConvTranspose blocks → RGB image

    Args:
        latent_dim: Dimension of latent space (default: 256)
        model_name: EfficientNet variant ('efficientnet_b0' to 'efficientnet_b7')
        pretrained: Whether to use pretrained encoder weights
        freeze_encoder: Whether to freeze encoder layers during training

    Example:
        >>> model = EfficientNetAutoencoder(latent_dim=256)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> reconstructed, latent = model(images)
        >>> print(reconstructed.shape, latent.shape)
        torch.Size([4, 3, 224, 224]) torch.Size([4, 256])
    """

    def __init__(
        self,
        latent_dim: int = 256,
        model_name: str = 'efficientnet_b0',
        pretrained: bool = True,
        freeze_encoder: bool = False
    ):
        super(EfficientNetAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.model_name = model_name

        # Load pretrained EfficientNet and get feature dimension
        encoder_dim, self.encoder = self._build_encoder(model_name, pretrained)

        # Bottleneck: Encoder features → Latent space
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_encode = nn.Linear(encoder_dim, latent_dim)

        # Decoder: Latent space → Image reconstruction
        self.fc_decode = nn.Linear(latent_dim, encoder_dim * 7 * 7)
        self.decoder = self._build_decoder(encoder_dim)

        # Initialize weights
        self._init_weights()

        # Optionally freeze encoder
        if freeze_encoder:
            self.freeze_encoder()

    def _build_encoder(
        self,
        model_name: str,
        pretrained: bool
    ) -> Tuple[int, nn.Module]:
        """
        Build EfficientNet encoder

        Returns:
            Tuple of (feature_dimension, encoder_module)
        """
        # Map model names to torchvision functions and feature dimensions
        model_configs = {
            'efficientnet_b0': (1280, models.efficientnet_b0),
            'efficientnet_b1': (1280, models.efficientnet_b1),
            'efficientnet_b2': (1408, models.efficientnet_b2),
            'efficientnet_b3': (1536, models.efficientnet_b3),
            'efficientnet_b4': (1792, models.efficientnet_b4),
            'efficientnet_b5': (2048, models.efficientnet_b5),
            'efficientnet_b6': (2304, models.efficientnet_b6),
            'efficientnet_b7': (2560, models.efficientnet_b7),
        }

        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_configs.keys())}")

        feature_dim, model_fn = model_configs[model_name]

        # Load model with appropriate weights
        if pretrained:
            # Get default weights for the model
            weights = 'IMAGENET1K_V1'
            encoder = model_fn(weights=weights)
        else:
            encoder = model_fn(weights=None)

        # Remove classifier head, keep only feature extractor
        encoder = encoder.features

        return feature_dim, encoder

    def _build_decoder(self, encoder_dim: int) -> nn.Module:
        """
        Build decoder with transposed convolutions

        Architecture:
            Input: (batch, encoder_dim, 7, 7)
            Block 1: (encoder_dim → encoder_dim//2) → (14, 14)
            Block 2: (encoder_dim//2 → encoder_dim//4) → (28, 28)
            Block 3: (encoder_dim//4 → encoder_dim//8) → (56, 56)
            Block 4: (encoder_dim//8 → 40) → (112, 112)
            Block 5: (40 → 3) → (224, 224)
            Output: (batch, 3, 224, 224)
        """
        decoder = nn.Sequential(
            # Block 1: 7×7 → 14×14
            ConvTransposeBlock(
                encoder_dim,
                encoder_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Block 2: 14×14 → 28×28
            ConvTransposeBlock(
                encoder_dim // 2,
                encoder_dim // 4,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Block 3: 28×28 → 56×56
            ConvTransposeBlock(
                encoder_dim // 4,
                encoder_dim // 8,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Block 4: 56×56 → 112×112
            ConvTransposeBlock(
                encoder_dim // 8,
                40,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Block 5: 112×112 → 224×224 (final layer, no BatchNorm/ReLU)
            nn.ConvTranspose2d(
                40,
                3,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Sigmoid()  # Output in [0, 1] range
        )

        return decoder

    def _init_weights(self):
        """Initialize decoder weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space

        Args:
            x: Input images (batch, 3, 224, 224)

        Returns:
            Latent representation (batch, latent_dim)
        """
        # Extract features via EfficientNet encoder
        features = self.encoder(x)  # (batch, encoder_dim, 7, 7)

        # Global average pooling
        pooled = self.avgpool(features)  # (batch, encoder_dim, 1, 1)
        pooled = torch.flatten(pooled, 1)  # (batch, encoder_dim)

        # Project to latent space
        latent = self.fc_encode(pooled)  # (batch, latent_dim)

        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            Reconstructed image (batch, 3, 224, 224)
        """
        # Project from latent space
        features = self.fc_decode(z)  # (batch, encoder_dim * 7 * 7)

        # Reshape to spatial dimensions
        batch_size = z.size(0)
        encoder_dim = features.size(1) // (7 * 7)
        features = features.view(batch_size, encoder_dim, 7, 7)

        # Upsample via decoder
        reconstructed = self.decoder(features)  # (batch, 3, 224, 224)

        return reconstructed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder

        Args:
            x: Input images (batch, 3, 224, 224)

        Returns:
            Tuple of (reconstructed_images, latent_representation)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def freeze_encoder(self):
        """Freeze encoder weights for faster training"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"Encoder ({self.model_name}) frozen")

    def unfreeze_encoder(self):
        """Unfreeze encoder weights for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print(f"Encoder ({self.model_name}) unfrozen")


class EfficientNetVAE(nn.Module):
    """
    Variational Autoencoder (VAE) with EfficientNet encoder

    Unlike standard autoencoder, VAE learns a probabilistic latent space
    with KL divergence regularization for better generalization.

    Architecture:
        Encoder: EfficientNet → mu and logvar (reparameterization trick)
        Decoder: Same as EfficientNetAutoencoder

    Loss: MSE(reconstruction) + beta * KL_divergence

    Args:
        latent_dim: Dimension of latent space (default: 256)
        model_name: EfficientNet variant
        pretrained: Use pretrained encoder weights
        freeze_encoder: Freeze encoder during training

    Example:
        >>> model = EfficientNetVAE(latent_dim=256)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> reconstructed, mu, logvar = model(images)
        >>> print(reconstructed.shape, mu.shape, logvar.shape)
        torch.Size([4, 3, 224, 224]) torch.Size([4, 256]) torch.Size([4, 256])
    """

    def __init__(
        self,
        latent_dim: int = 256,
        model_name: str = 'efficientnet_b0',
        pretrained: bool = True,
        freeze_encoder: bool = False
    ):
        super(EfficientNetVAE, self).__init__()

        self.latent_dim = latent_dim
        self.model_name = model_name

        # Load pretrained EfficientNet
        encoder_dim, self.encoder = self._build_encoder(model_name, pretrained)

        # Bottleneck: Encoder features → Latent distribution parameters
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(encoder_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, encoder_dim * 7 * 7)
        self.decoder = self._build_decoder(encoder_dim)

        # Initialize weights
        self._init_weights()

        # Optionally freeze encoder
        if freeze_encoder:
            self.freeze_encoder()

    def _build_encoder(
        self,
        model_name: str,
        pretrained: bool
    ) -> Tuple[int, nn.Module]:
        """Build EfficientNet encoder (same as standard autoencoder)"""
        model_configs = {
            'efficientnet_b0': (1280, models.efficientnet_b0),
            'efficientnet_b1': (1280, models.efficientnet_b1),
            'efficientnet_b2': (1408, models.efficientnet_b2),
            'efficientnet_b3': (1536, models.efficientnet_b3),
            'efficientnet_b4': (1792, models.efficientnet_b4),
            'efficientnet_b5': (2048, models.efficientnet_b5),
            'efficientnet_b6': (2304, models.efficientnet_b6),
            'efficientnet_b7': (2560, models.efficientnet_b7),
        }

        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}")

        feature_dim, model_fn = model_configs[model_name]

        if pretrained:
            encoder = model_fn(weights='IMAGENET1K_V1')
        else:
            encoder = model_fn(weights=None)

        encoder = encoder.features

        return feature_dim, encoder

    def _build_decoder(self, encoder_dim: int) -> nn.Module:
        """Build decoder (same as standard autoencoder)"""
        decoder = nn.Sequential(
            ConvTransposeBlock(encoder_dim, encoder_dim // 2, 4, 2, 1),
            ConvTransposeBlock(encoder_dim // 2, encoder_dim // 4, 4, 2, 1),
            ConvTransposeBlock(encoder_dim // 4, encoder_dim // 8, 4, 2, 1),
            ConvTransposeBlock(encoder_dim // 8, 40, 4, 2, 1),
            nn.ConvTranspose2d(40, 3, 4, 2, 1),
            nn.Sigmoid()
        )
        return decoder

    def _init_weights(self):
        """Initialize decoder weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution parameters

        Args:
            x: Input images (batch, 3, 224, 224)

        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        # Extract features
        features = self.encoder(x)  # (batch, encoder_dim, 7, 7)

        # Global average pooling
        pooled = self.avgpool(features)
        pooled = torch.flatten(pooled, 1)

        # Get distribution parameters
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)

        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log variance of latent distribution (batch, latent_dim)

        Returns:
            Sampled latent vector (batch, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            Reconstructed image (batch, 3, 224, 224)
        """
        features = self.fc_decode(z)

        batch_size = z.size(0)
        encoder_dim = features.size(1) // (7 * 7)
        features = features.view(batch_size, encoder_dim, 7, 7)

        reconstructed = self.decoder(features)

        return reconstructed

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE

        Args:
            x: Input images (batch, 3, 224, 224)

        Returns:
            Tuple of (reconstructed, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def freeze_encoder(self):
        """Freeze encoder weights"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"Encoder ({self.model_name}) frozen")

    def unfreeze_encoder(self):
        """Unfreeze encoder weights"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print(f"Encoder ({self.model_name}) unfrozen")


def reconstruction_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute reconstruction loss (MSE)

    Args:
        reconstructed: Reconstructed images (batch, 3, H, W)
        original: Original images (batch, 3, H, W)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Reconstruction loss
    """
    return F.mse_loss(reconstructed, original, reduction=reduction)


def gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Compute gradient loss to preserve edges and texture details

    Computes gradients in both x and y directions and compares them
    between predicted and target images.

    Args:
        pred: Predicted/reconstructed images (batch, C, H, W)
        target: Target/original images (batch, C, H, W)

    Returns:
        Gradient loss value

    Example:
        >>> grad_loss = gradient_loss(reconstructed, original)
    """
    # Compute gradients in x direction (horizontal)
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    # Compute gradients in y direction (vertical)
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    # Compute MSE on gradients
    loss_dx = F.mse_loss(pred_dx, target_dx)
    loss_dy = F.mse_loss(pred_dy, target_dy)

    return loss_dx + loss_dy


def combined_reconstruction_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    perceptual_loss_fn: Optional[PerceptualLoss] = None,
    perceptual_weight: float = 0.1,
    gradient_weight: float = 0.05,
    mse_weight: float = 1.0
) -> Tuple[torch.Tensor, dict]:
    """
    Combined reconstruction loss with MSE, perceptual, and gradient components

    Args:
        reconstructed: Reconstructed images (batch, 3, H, W)
        original: Original images (batch, 3, H, W)
        perceptual_loss_fn: Optional PerceptualLoss instance
        perceptual_weight: Weight for perceptual loss component (default: 0.1)
        gradient_weight: Weight for gradient loss component (default: 0.05)
        mse_weight: Weight for MSE loss component (default: 1.0, set to 0.0 to disable)

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains individual components

    Example:
        >>> perceptual_fn = PerceptualLoss(device='cuda')
        >>> loss, components = combined_reconstruction_loss(
        ...     reconstructed, original, perceptual_fn,
        ...     perceptual_weight=0.1, gradient_weight=0.05
        ... )
    """
    # MSE loss
    mse_loss = F.mse_loss(reconstructed, original)

    # Initialize components dict
    loss_components = {
        'mse': mse_loss.item(),
        'perceptual': 0.0,
        'gradient': 0.0
    }

    # Start with MSE loss (weighted)
    total_loss = mse_weight * mse_loss

    # Add perceptual loss if available
    if perceptual_loss_fn is not None:
        perc_loss = perceptual_loss_fn(reconstructed, original)
        total_loss = total_loss + perceptual_weight * perc_loss
        loss_components['perceptual'] = perc_loss.item()

    # Add gradient loss
    grad_loss = gradient_loss(reconstructed, original)
    total_loss = total_loss + gradient_weight * grad_loss
    loss_components['gradient'] = grad_loss.item()

    return total_loss, loss_components


def vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    perceptual_loss_fn: Optional[PerceptualLoss] = None,
    perceptual_weight: float = 0.1,
    gradient_weight: float = 0.05,
    mse_weight: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Compute VAE loss: reconstruction + KL divergence + perceptual + gradient

    Args:
        reconstructed: Reconstructed images
        original: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence (beta-VAE)
        perceptual_loss_fn: Optional PerceptualLoss instance
        perceptual_weight: Weight for perceptual loss component (default: 0.1)
        gradient_weight: Weight for gradient loss component (default: 0.05)
        mse_weight: Weight for MSE loss component (default: 1.0, set to 0.0 to disable)

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence, loss_components_dict)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed, original, reduction='mean')

    # Initialize components dict
    loss_components = {
        'mse': recon_loss.item(),
        'perceptual': 0.0,
        'gradient': 0.0
    }

    # Start with reconstruction loss (weighted)
    total_recon = mse_weight * recon_loss

    # Add perceptual loss if available
    if perceptual_loss_fn is not None:
        perc_loss = perceptual_loss_fn(reconstructed, original)
        total_recon = total_recon + perceptual_weight * perc_loss
        loss_components['perceptual'] = perc_loss.item()

    # Add gradient loss
    grad_loss = gradient_loss(reconstructed, original)
    total_recon = total_recon + gradient_weight * grad_loss
    loss_components['gradient'] = grad_loss.item()

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_div = kl_div.mean()

    # Total loss
    total_loss = total_recon + beta * kl_div

    return total_loss, total_recon, kl_div, loss_components


def get_model(
    model_type: str = 'standard',
    latent_dim: int = 256,
    model_name: str = 'efficientnet_b0',
    pretrained: bool = True,
    freeze_encoder: bool = False
):
    """
    Factory function to create autoencoder models

    Args:
        model_type: 'standard' or 'vae'
        latent_dim: Latent space dimension
        model_name: EfficientNet variant
        pretrained: Use pretrained encoder
        freeze_encoder: Freeze encoder layers

    Returns:
        Autoencoder model instance

    Example:
        >>> model = get_model('standard', latent_dim=256)
        >>> model = get_model('vae', latent_dim=512, model_name='efficientnet_b1')
    """
    if model_type == 'standard':
        return EfficientNetAutoencoder(
            latent_dim=latent_dim,
            model_name=model_name,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder
        )
    elif model_type in ['vae', 'variational']:
        return EfficientNetVAE(
            latent_dim=latent_dim,
            model_name=model_name,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'standard' or 'vae'")


if __name__ == '__main__':
    # Test standard autoencoder
    print("=" * 80)
    print("Testing EfficientNetAutoencoder")
    print("=" * 80)

    model = EfficientNetAutoencoder(latent_dim=256)
    x = torch.randn(4, 3, 224, 224)

    reconstructed, latent = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent shape: {latent.shape}")

    # Test VAE
    print("\n" + "=" * 80)
    print("Testing EfficientNetVAE")
    print("=" * 80)

    vae_model = EfficientNetVAE(latent_dim=256)
    reconstructed, mu, logvar = vae_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # Test loss functions
    print("\n" + "=" * 80)
    print("Testing Loss Functions")
    print("=" * 80)

    recon_loss = reconstruction_loss(reconstructed, x)
    print(f"Reconstruction loss: {recon_loss.item():.6f}")

    total, recon, kl = vae_loss(reconstructed, x, mu, logvar, beta=1.0)
    print(f"VAE total loss: {total.item():.6f}")
    print(f"VAE recon loss: {recon.item():.6f}")
    print(f"VAE KL divergence: {kl.item():.6f}")
