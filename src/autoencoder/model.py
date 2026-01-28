"""
Autoencoder model for end-grain feature compression

This module implements a deep autoencoder that learns to compress
the 265-dimensional feature vector into a lower-dimensional latent space,
enabling efficient representation learning for end-grain detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FeatureAutoencoder(nn.Module):
    """
    Deep autoencoder for end-grain feature compression

    Architecture:
        Encoder: 265 -> 128 -> 64 -> 32 -> latent_dim
        Decoder: latent_dim -> 32 -> 64 -> 128 -> 265

    Args:
        input_dim: Input feature dimension (default: 265)
        latent_dim: Latent space dimension (default: 16)
        hidden_dims: List of hidden layer dimensions (default: [128, 64, 32])
        dropout_rate: Dropout probability (default: 0.2)
        activation: Activation function ('relu', 'leaky_relu', 'elu')

    Example:
        >>> model = FeatureAutoencoder(input_dim=265, latent_dim=16)
        >>> features = torch.randn(32, 265)  # Batch of 32 samples
        >>> reconstructed, latent = model(features)
        >>> print(reconstructed.shape, latent.shape)
        torch.Size([32, 265]) torch.Size([32, 16])
    """

    def __init__(
        self,
        input_dim: int = 265,
        latent_dim: int = 16,
        hidden_dims: list = [128, 64, 32],
        dropout_rate: float = 0.2,
        activation: str = 'relu'
    ):
        super(FeatureAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build encoder
        self.encoder = self._build_encoder()

        # Build decoder
        self.decoder = self._build_decoder()

        # Initialize weights
        self.apply(self._init_weights)

    def _build_encoder(self) -> nn.Sequential:
        """Build encoder network"""
        layers = []

        # Input layer
        prev_dim = self.input_dim

        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim

        # Bottleneck layer (no activation/dropout)
        layers.append(nn.Linear(prev_dim, self.latent_dim))

        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        """Build decoder network (mirror of encoder)"""
        layers = []

        # Start from latent dimension
        prev_dim = self.latent_dim

        # Hidden layers (reverse order of encoder)
        for hidden_dim in reversed(self.hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer (no activation - regression task)
        layers.append(nn.Linear(prev_dim, self.input_dim))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode features to latent space

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Latent representation (batch_size, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to features

        Args:
            z: Latent representation (batch_size, latent_dim)

        Returns:
            Reconstructed features (batch_size, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Tuple of (reconstructed_features, latent_representation)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for end-grain feature compression

    Unlike standard autoencoder, VAE learns a probabilistic latent space
    with regularization via KL divergence, enabling better generalization
    and sampling capabilities.

    Args:
        input_dim: Input feature dimension (default: 265)
        latent_dim: Latent space dimension (default: 16)
        hidden_dims: List of hidden layer dimensions (default: [128, 64, 32])
        dropout_rate: Dropout probability (default: 0.2)

    Example:
        >>> model = VariationalAutoencoder(input_dim=265, latent_dim=16)
        >>> features = torch.randn(32, 265)
        >>> reconstructed, mu, logvar = model(features)
    """

    def __init__(
        self,
        input_dim: int = 265,
        latent_dim: int = 16,
        hidden_dims: list = [128, 64, 32],
        dropout_rate: float = 0.2
    ):
        super(VariationalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode to latent distribution parameters

        Returns:
            Tuple of (mu, logvar)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Returns:
            Tuple of (reconstructed, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


def vae_loss(reconstructed, original, mu, logvar, beta=1.0):
    """
    VAE loss function: reconstruction loss + KL divergence

    Args:
        reconstructed: Reconstructed features
        original: Original features
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (beta-VAE)

    Returns:
        Total loss, reconstruction loss, KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed, original, reduction='mean')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_div = kl_div.mean()

    # Total loss
    total_loss = recon_loss + beta * kl_div

    return total_loss, recon_loss, kl_div


def get_model(model_type='standard', **kwargs):
    """
    Factory function to create autoencoder models

    Args:
        model_type: 'standard' or 'variational'
        **kwargs: Model parameters

    Returns:
        Autoencoder model instance
    """
    if model_type == 'standard':
        return FeatureAutoencoder(**kwargs)
    elif model_type == 'variational':
        return VariationalAutoencoder(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
