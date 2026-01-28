"""
VGG-based Autoencoders for One-Class Anomaly Detection

This module implements VGG16/19-based autoencoders for end-grain wood detection
using one-class classification via reconstruction error.

Architecture:
    - Encoder: Pretrained VGG16/19 (from ImageNet)
    - Decoder: Symmetric transposed convolutions for image reconstruction
    - Training: Unsupervised on end-grain images only
    - Inference: Low reconstruction error = end-grain, High error = world (outlier)

Research Validation:
    - VGG-16 Autoencoder for Retinal Anomaly Detection (ACM 2024)
    - Heritage Building Roof Anomaly Detection (MDPI 2024)
    - Deep Perceptual Autoencoders for Medical Imaging (ArXiv 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg16, vgg19, VGG16_Weights, VGG19_Weights
from typing import Tuple, Optional


class VGGConvTransposeBlock(nn.Module):
    """
    Transposed convolution block for VGG decoder upsampling

    Architecture: ConvTranspose2d → Normalization → ReLU → Optional extra Conv2d layers

    This block mimics VGG's stacked convolution style with:
    - Learnable upsampling via ConvTranspose2d
    - Configurable normalization (None, Batch, Instance, Group)
    - Optional extra convolution layers for increased depth

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for ConvTranspose2d (default: 4)
        stride: Stride for upsampling (default: 2)
        padding: Padding for ConvTranspose2d (default: 1)
        norm_type: Normalization type: 'none', 'batch', 'instance', 'group'
        num_extra_convs: Number of additional 3×3 convolutions after upsample (default: 0)

    Example:
        >>> block = VGGConvTransposeBlock(512, 256, norm_type='instance', num_extra_convs=2)
        >>> x = torch.randn(4, 512, 7, 7)
        >>> out = block(x)  # (4, 256, 14, 14)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm_type: str = 'instance',
        num_extra_convs: int = 0
    ):
        super(VGGConvTransposeBlock, self).__init__()

        layers = []

        # Transposed convolution for upsampling
        layers.append(nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(norm_type == 'none')
        ))

        # Normalization
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm_type == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif norm_type == 'group':
            layers.append(nn.GroupNorm(32, out_channels))
        elif norm_type != 'none':
            raise ValueError(f"Unknown norm_type: {norm_type}. Choose from 'none', 'batch', 'instance', 'group'")

        # Activation
        layers.append(nn.ReLU(inplace=True))

        self.upsample = nn.Sequential(*layers)

        # Extra convolution layers (VGG style stacked convs)
        self.extra_convs = None
        if num_extra_convs > 0:
            extra_layers = []
            for _ in range(num_extra_convs):
                extra_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=(norm_type == 'none')))

                if norm_type == 'batch':
                    extra_layers.append(nn.BatchNorm2d(out_channels))
                elif norm_type == 'instance':
                    extra_layers.append(nn.InstanceNorm2d(out_channels, affine=True))
                elif norm_type == 'group':
                    extra_layers.append(nn.GroupNorm(32, out_channels))

                extra_layers.append(nn.ReLU(inplace=True))

            self.extra_convs = nn.Sequential(*extra_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through upsample block"""
        x = self.upsample(x)
        if self.extra_convs is not None:
            x = self.extra_convs(x)
        return x


class VGGAutoencoder(nn.Module):
    """
    VGG-based autoencoder for one-class anomaly detection

    Trains on end-grain images only and classifies via reconstruction error:
    - Low reconstruction error → End-grain (inlier)
    - High reconstruction error → World/Non-wood (outlier)

    Architecture:
        Encoder: VGG16/19 (pretrained) → 512 features at 7×7 spatial resolution
        Bottleneck: AdaptiveAvgPool → Linear(512 → latent_dim)
        Decoder: Linear(latent_dim → 512×7×7) → 5 ConvTranspose blocks → RGB image

    Args:
        latent_dim: Dimension of latent space (default: 512)
        vgg_variant: 'vgg16' or 'vgg19' (default: 'vgg16')
        pretrained: Whether to use pretrained encoder weights (default: True)
        freeze_encoder: Whether to freeze encoder layers during training (default: True)
        decoder_norm_type: Normalization type for decoder: 'none', 'batch', 'instance', 'group' (default: 'instance')

    Example:
        >>> model = VGGAutoencoder(latent_dim=512, vgg_variant='vgg16')
        >>> images = torch.randn(4, 3, 224, 224)
        >>> reconstructed, latent = model(images)
        >>> print(reconstructed.shape, latent.shape)
        torch.Size([4, 3, 224, 224]) torch.Size([4, 512])
    """

    def __init__(
        self,
        latent_dim: int = 512,
        vgg_variant: str = 'vgg16',
        pretrained: bool = True,
        freeze_encoder: bool = True,
        decoder_norm_type: str = 'instance'
    ):
        super(VGGAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.vgg_variant = vgg_variant
        self.decoder_norm_type = decoder_norm_type

        # Build VGG encoder
        self.vgg_encoder = self._build_encoder(vgg_variant, pretrained)

        # Bottleneck: 512 channels → latent space
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_encode = nn.Linear(512, latent_dim)

        # Decoder: latent space → 512×7×7 → image reconstruction
        self.fc_decode = nn.Linear(latent_dim, 512 * 7 * 7)
        self.decoder = self._build_decoder(decoder_norm_type)

        # Initialize decoder weights
        self._init_decoder_weights()

        # Optionally freeze encoder
        if freeze_encoder:
            self.freeze_encoder()

    def _build_encoder(self, vgg_variant: str, pretrained: bool) -> nn.Module:
        """
        Build VGG encoder (features only, no classifier)

        Args:
            vgg_variant: 'vgg16' or 'vgg19'
            pretrained: Whether to load ImageNet pretrained weights

        Returns:
            VGG feature extractor (5 blocks up to pool5)
        """
        if vgg_variant == 'vgg16':
            if pretrained:
                vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            else:
                vgg_model = vgg16(weights=None)
        elif vgg_variant == 'vgg19':
            if pretrained:
                vgg_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            else:
                vgg_model = vgg19(weights=None)
        else:
            raise ValueError(f"Unknown vgg_variant: {vgg_variant}. Choose 'vgg16' or 'vgg19'")

        # Extract feature layers (up to pool5, output: 512×7×7)
        encoder = vgg_model.features

        return encoder

    def _build_decoder(self, norm_type: str) -> nn.Module:
        """
        Build symmetric decoder with 5 upsample blocks

        Architecture mirrors VGG encoder in reverse:
            Block 5: 7×7 → 14×14 (512 → 512 channels, 2 extra convs)
            Block 4: 14×14 → 28×28 (512 → 256 channels, 2 extra convs)
            Block 3: 28×28 → 56×56 (256 → 128 channels, 1 extra conv)
            Block 2: 56×56 → 112×112 (128 → 64 channels, 1 extra conv)
            Block 1: 112×112 → 224×224 (64 → 3 channels, final output)

        Args:
            norm_type: Normalization type for decoder blocks

        Returns:
            Decoder module
        """
        decoder = nn.Sequential(
            # Block 5: 7×7 → 14×14 (512 → 512, 2 extra convs for VGG depth)
            VGGConvTransposeBlock(512, 512, kernel_size=4, stride=2, padding=1,
                                  norm_type=norm_type, num_extra_convs=2),

            # Block 4: 14×14 → 28×28 (512 → 256, 2 extra convs)
            VGGConvTransposeBlock(512, 256, kernel_size=4, stride=2, padding=1,
                                  norm_type=norm_type, num_extra_convs=2),

            # Block 3: 28×28 → 56×56 (256 → 128, 1 extra conv)
            VGGConvTransposeBlock(256, 128, kernel_size=4, stride=2, padding=1,
                                  norm_type=norm_type, num_extra_convs=1),

            # Block 2: 56×56 → 112×112 (128 → 64, 1 extra conv)
            VGGConvTransposeBlock(128, 64, kernel_size=4, stride=2, padding=1,
                                  norm_type=norm_type, num_extra_convs=1),

            # Block 1: 112×112 → 224×224 (64 → 64, 1 extra conv, then final to 3 channels)
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

        return decoder

    def _init_decoder_weights(self):
        """Initialize decoder weights with Kaiming initialization"""
        for m in self.decoder.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize bottleneck layers
        nn.init.kaiming_normal_(self.fc_encode.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc_encode.bias, 0)
        nn.init.kaiming_normal_(self.fc_decode.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc_decode.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space

        Args:
            x: Input images (batch, 3, 224, 224)

        Returns:
            Latent representation (batch, latent_dim)
        """
        # Extract VGG features
        features = self.vgg_encoder(x)  # (batch, 512, 7, 7)

        # Global average pooling
        pooled = self.avgpool(features)  # (batch, 512, 1, 1)
        pooled = torch.flatten(pooled, 1)  # (batch, 512)

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
        features = self.fc_decode(z)  # (batch, 512 * 7 * 7)

        # Reshape to spatial dimensions
        batch_size = z.size(0)
        features = features.view(batch_size, 512, 7, 7)

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
        """Freeze VGG encoder weights for faster training"""
        for param in self.vgg_encoder.parameters():
            param.requires_grad = False
        print(f"VGG encoder ({self.vgg_variant}) frozen")

    def unfreeze_encoder(self):
        """Unfreeze VGG encoder weights for fine-tuning"""
        for param in self.vgg_encoder.parameters():
            param.requires_grad = True
        print(f"VGG encoder ({self.vgg_variant}) unfrozen for fine-tuning")


class VGGVAE(nn.Module):
    """
    Variational Autoencoder (VAE) with VGG encoder

    Unlike standard autoencoder, VAE learns a probabilistic latent space
    with KL divergence regularization for better generalization.

    Architecture:
        Encoder: VGG16/19 → mu and logvar (reparameterization trick)
        Decoder: Same as VGGAutoencoder

    Loss: MSE(reconstruction) + beta * KL_divergence

    Args:
        latent_dim: Dimension of latent space (default: 512)
        vgg_variant: 'vgg16' or 'vgg19' (default: 'vgg16')
        pretrained: Use pretrained encoder weights (default: True)
        freeze_encoder: Freeze encoder during training (default: True)
        decoder_norm_type: Normalization type for decoder (default: 'instance')

    Example:
        >>> model = VGGVAE(latent_dim=512, vgg_variant='vgg16')
        >>> images = torch.randn(4, 3, 224, 224)
        >>> reconstructed, mu, logvar = model(images)
        >>> print(reconstructed.shape, mu.shape, logvar.shape)
        torch.Size([4, 3, 224, 224]) torch.Size([4, 512]) torch.Size([4, 512])
    """

    def __init__(
        self,
        latent_dim: int = 512,
        vgg_variant: str = 'vgg16',
        pretrained: bool = True,
        freeze_encoder: bool = True,
        decoder_norm_type: str = 'instance'
    ):
        super(VGGVAE, self).__init__()

        self.latent_dim = latent_dim
        self.vgg_variant = vgg_variant
        self.decoder_norm_type = decoder_norm_type

        # Build VGG encoder
        self.vgg_encoder = self._build_encoder(vgg_variant, pretrained)

        # Bottleneck: 512 channels → latent distribution parameters
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder: latent space → image reconstruction
        self.fc_decode = nn.Linear(latent_dim, 512 * 7 * 7)
        self.decoder = self._build_decoder(decoder_norm_type)

        # Initialize decoder weights
        self._init_decoder_weights()

        # Optionally freeze encoder
        if freeze_encoder:
            self.freeze_encoder()

    def _build_encoder(self, vgg_variant: str, pretrained: bool) -> nn.Module:
        """Build VGG encoder (same as VGGAutoencoder)"""
        if vgg_variant == 'vgg16':
            if pretrained:
                vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            else:
                vgg_model = vgg16(weights=None)
        elif vgg_variant == 'vgg19':
            if pretrained:
                vgg_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            else:
                vgg_model = vgg19(weights=None)
        else:
            raise ValueError(f"Unknown vgg_variant: {vgg_variant}. Choose 'vgg16' or 'vgg19'")

        encoder = vgg_model.features
        return encoder

    def _build_decoder(self, norm_type: str) -> nn.Module:
        """Build decoder (same as VGGAutoencoder)"""
        decoder = nn.Sequential(
            VGGConvTransposeBlock(512, 512, kernel_size=4, stride=2, padding=1,
                                  norm_type=norm_type, num_extra_convs=2),
            VGGConvTransposeBlock(512, 256, kernel_size=4, stride=2, padding=1,
                                  norm_type=norm_type, num_extra_convs=2),
            VGGConvTransposeBlock(256, 128, kernel_size=4, stride=2, padding=1,
                                  norm_type=norm_type, num_extra_convs=1),
            VGGConvTransposeBlock(128, 64, kernel_size=4, stride=2, padding=1,
                                  norm_type=norm_type, num_extra_convs=1),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        return decoder

    def _init_decoder_weights(self):
        """Initialize decoder weights"""
        for m in self.decoder.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize bottleneck layers
        nn.init.kaiming_normal_(self.fc_mu.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc_mu.bias, 0)
        nn.init.kaiming_normal_(self.fc_logvar.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc_logvar.bias, 0)
        nn.init.kaiming_normal_(self.fc_decode.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc_decode.bias, 0)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution parameters

        Args:
            x: Input images (batch, 3, 224, 224)

        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        # Extract VGG features
        features = self.vgg_encoder(x)  # (batch, 512, 7, 7)

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
        features = features.view(batch_size, 512, 7, 7)
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
        """Freeze VGG encoder weights"""
        for param in self.vgg_encoder.parameters():
            param.requires_grad = False
        print(f"VGG encoder ({self.vgg_variant}) frozen")

    def unfreeze_encoder(self):
        """Unfreeze VGG encoder weights"""
        for param in self.vgg_encoder.parameters():
            param.requires_grad = True
        print(f"VGG encoder ({self.vgg_variant}) unfrozen for fine-tuning")


def get_vgg_model(
    model_type: str = 'standard',
    latent_dim: int = 512,
    vgg_variant: str = 'vgg16',
    pretrained: bool = True,
    freeze_encoder: bool = True,
    decoder_norm_type: str = 'instance'
):
    """
    Factory function to create VGG-based autoencoder models

    Args:
        model_type: 'standard' or 'vae'
        latent_dim: Latent space dimension (default: 512)
        vgg_variant: 'vgg16' or 'vgg19' (default: 'vgg16')
        pretrained: Use pretrained encoder (default: True)
        freeze_encoder: Freeze encoder layers (default: True)
        decoder_norm_type: Decoder normalization: 'none', 'batch', 'instance', 'group' (default: 'instance')

    Returns:
        VGG autoencoder model instance

    Example:
        >>> model = get_vgg_model('standard', latent_dim=512, vgg_variant='vgg16')
        >>> model = get_vgg_model('vae', latent_dim=512, vgg_variant='vgg19')
    """
    if model_type == 'standard':
        return VGGAutoencoder(
            latent_dim=latent_dim,
            vgg_variant=vgg_variant,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            decoder_norm_type=decoder_norm_type
        )
    elif model_type in ['vae', 'variational']:
        return VGGVAE(
            latent_dim=latent_dim,
            vgg_variant=vgg_variant,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            decoder_norm_type=decoder_norm_type
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'standard' or 'vae'")


if __name__ == '__main__':
    # Test VGG16 autoencoder
    print("=" * 80)
    print("Testing VGGAutoencoder (VGG16)")
    print("=" * 80)

    model = VGGAutoencoder(latent_dim=512, vgg_variant='vgg16')
    x = torch.randn(2, 3, 224, 224)

    reconstructed, latent = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent shape: {latent.shape}")

    # Test VGG19 VAE
    print("\n" + "=" * 80)
    print("Testing VGGVAE (VGG19)")
    print("=" * 80)

    vae_model = VGGVAE(latent_dim=512, vgg_variant='vgg19')
    reconstructed, mu, logvar = vae_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    print("\n" + "=" * 80)
    print("Tests Completed Successfully!")
    print("=" * 80)
