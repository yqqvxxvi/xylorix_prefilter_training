"""
Contrastive Learning Module

Self-supervised learning using contrastive methods (SimCLR, etc.)
for wood classification.
"""

from .augmentations import (
    get_contrastive_augmentation,
    get_simclr_augmentation_pair,
    ContrastiveTransformations,
    GaussianBlur,
    Solarize
)

from .losses import (
    NTXentLoss,
    SimCLRLoss,
    SupConLoss
)

from .model import (
    SimCLRModel,
    ProjectionHead,
    LinearClassifier
)

from .dataset import (
    ContrastiveDataset,
    ContrastiveDatasetWrapper,
    contrastive_collate_fn
)

__all__ = [
    # Augmentations
    'get_contrastive_augmentation',
    'get_simclr_augmentation_pair',
    'ContrastiveTransformations',
    'GaussianBlur',
    'Solarize',
    # Losses
    'NTXentLoss',
    'SimCLRLoss',
    'SupConLoss',
    # Models
    'SimCLRModel',
    'ProjectionHead',
    'LinearClassifier',
    # Dataset
    'ContrastiveDataset',
    'ContrastiveDatasetWrapper',
    'contrastive_collate_fn',
]
