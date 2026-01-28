"""
End-grain wood autoencoder training

This module provides both:
1. Feature-based autoencoders (handcrafted features → latent space)
2. Convolutional autoencoders (images → latent space) for one-class anomaly detection

RECOMMENDED: Use convolutional autoencoders (conv_model) for better performance
"""

# ============================================================================
# Legacy: Feature-based Autoencoders (handcrafted features)
# ============================================================================
from .feature_extractor import EndGrainFeatureExtractor
from .visualizer import FeatureVisualizer
from .model import (
    FeatureAutoencoder,
    VariationalAutoencoder,
    vae_loss as feature_vae_loss
)

# ============================================================================
# NEW: Convolutional Autoencoders (deep learning, one-class anomaly detection)
# ============================================================================
from .conv_model import (
    EfficientNetAutoencoder,
    EfficientNetVAE,
    ConvTransposeBlock,
    reconstruction_loss,
    vae_loss,
    get_model
)

# ============================================================================
# Dataset Classes
# ============================================================================
from .dataset import (
    # Feature-based datasets
    FeatureDataset,
    PrecomputedFeatureDataset,
    create_endgrain_world_dataset,

    # Image-based datasets (for convolutional autoencoders)
    ImageAutoencoderDataset,
    create_oneclass_dataset,
    create_anomaly_test_dataset,

    # Common utilities
    create_data_loaders,
    load_image_paths_from_directory
)

__all__ = [
    # Legacy feature-based autoencoders
    'EndGrainFeatureExtractor',
    'FeatureVisualizer',
    'FeatureAutoencoder',
    'VariationalAutoencoder',
    'feature_vae_loss',

    # NEW: Convolutional autoencoders
    'EfficientNetAutoencoder',
    'EfficientNetVAE',
    'ConvTransposeBlock',
    'reconstruction_loss',
    'vae_loss',
    'get_model',

    # Feature datasets
    'FeatureDataset',
    'PrecomputedFeatureDataset',
    'create_endgrain_world_dataset',

    # Image datasets
    'ImageAutoencoderDataset',
    'create_oneclass_dataset',
    'create_anomaly_test_dataset',

    # Common utilities
    'create_data_loaders',
    'load_image_paths_from_directory'
]
