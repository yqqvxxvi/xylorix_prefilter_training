"""
Models for contrastive learning

Implements encoder with projection head for self-supervised learning.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class ProjectionHead(nn.Module):
    """
    Projection head (MLP) for contrastive learning

    The projection head is a small MLP applied after the encoder.
    SimCLR paper found that adding a projection head significantly improves
    the quality of learned representations.

    Architecture: Linear → BatchNorm → ReLU → Linear

    After training, the projection head is typically discarded, and only
    the encoder is used for downstream tasks.
    """

    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 2048,
                 output_dim: int = 128):
        """
        Args:
            input_dim: Dimension of encoder output (e.g., 2048 for ResNet50)
            hidden_dim: Dimension of hidden layer (SimCLR uses same as input_dim)
            output_dim: Dimension of projection output (128 for SimCLR)
        """
        super(ProjectionHead, self).__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder output (batch_size, input_dim)

        Returns:
            Projected embedding (batch_size, output_dim)
        """
        return self.projection(x)


class SimCLRModel(nn.Module):
    """
    SimCLR model: Encoder + Projection Head

    This model combines a backbone encoder (e.g., ResNet) with a projection head
    for self-supervised contrastive learning.

    Usage:
        1. Pre-training: Use full model (encoder + projection head) with contrastive loss
        2. Fine-tuning: Remove projection head, use encoder.get_representation()
                        Add task-specific head for classification/regression
    """

    def __init__(self,
                 encoder_name: str = 'resnet50',
                 pretrained: bool = False,
                 projection_dim: int = 128,
                 hidden_dim: Optional[int] = None,
                 grayscale: bool = False):
        """
        Args:
            encoder_name: Backbone encoder ('resnet18', 'resnet50', 'efficientnet_b0', etc.)
            pretrained: Whether to use ImageNet pretrained weights (usually False for SSL)
            projection_dim: Output dimension of projection head (default 128)
            hidden_dim: Hidden dimension of projection head (default: same as encoder output)
            grayscale: If True, modify first conv layer for 1-channel input
        """
        super(SimCLRModel, self).__init__()

        # Create encoder
        self.encoder_name = encoder_name
        self.encoder, encoder_dim = self._build_encoder(encoder_name, pretrained, grayscale)

        # Create projection head
        if hidden_dim is None:
            hidden_dim = encoder_dim
        self.projection_head = ProjectionHead(encoder_dim, hidden_dim, projection_dim)

        self.encoder_dim = encoder_dim
        self.projection_dim = projection_dim

    def _build_encoder(self, name: str, pretrained: bool, grayscale: bool) -> Tuple[nn.Module, int]:
        """
        Build encoder backbone

        Returns:
            (encoder, output_dimension)
        """
        if name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            encoder_dim = 512
        elif name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            encoder_dim = 2048
        elif name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
            encoder_dim = 2048
        elif name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            encoder_dim = 1280
        else:
            raise ValueError(f"Unknown encoder: {name}")

        # Modify for grayscale if needed
        if grayscale:
            if 'resnet' in name:
                # Modify first conv layer for 1-channel input
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            elif 'efficientnet' in name:
                # Modify first conv layer for EfficientNet
                original_conv = model.features[0][0]
                model.features[0][0] = nn.Conv2d(
                    1, original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False
                )

        # Remove classifier head
        if 'resnet' in name:
            # Remove final FC layer, keep avgpool
            encoder = nn.Sequential(*list(model.children())[:-1])
        elif 'efficientnet' in name:
            # Remove classifier, keep avgpool
            encoder = nn.Sequential(
                model.features,
                model.avgpool,
                nn.Flatten()
            )
        else:
            encoder = model

        return encoder, encoder_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input images (batch_size, channels, height, width)

        Returns:
            (representation, projection)
            - representation: Encoder output (batch_size, encoder_dim)
            - projection: Projection head output (batch_size, projection_dim)

        Note: During contrastive training, use 'projection' for computing loss.
              For downstream tasks, use 'representation'.
        """
        # Get encoder representation
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)  # Flatten if needed

        # Get projection
        z = self.projection_head(h)

        return h, z

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get encoder representation only (for downstream tasks)

        Args:
            x: Input images

        Returns:
            Encoder output (batch_size, encoder_dim)
        """
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return h

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get projection embedding (for contrastive learning)

        Args:
            x: Input images

        Returns:
            Projected embedding (batch_size, projection_dim)
        """
        _, z = self.forward(x)
        return z


class LinearClassifier(nn.Module):
    """
    Linear classifier for evaluating learned representations

    After pre-training with contrastive learning, evaluate the quality of
    learned representations by training a linear classifier on top.

    This is the standard evaluation protocol for self-supervised learning.
    """

    def __init__(self, input_dim: int, num_classes: int):
        """
        Args:
            input_dim: Dimension of input features (encoder output dim)
            num_classes: Number of classes
        """
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Class logits (batch_size, num_classes)
        """
        return self.fc(x)
