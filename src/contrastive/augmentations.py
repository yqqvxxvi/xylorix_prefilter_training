"""
Aggressive data augmentation for contrastive learning

Based on SimCLR (A Simple Framework for Contrastive Learning of Visual Representations)
and recent best practices for self-supervised contrastive learning.

These augmentations are MORE aggressive than standard supervised learning to ensure
the model learns robust, invariant representations.
"""

import torch
from torchvision import transforms
from typing import Tuple
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps


class GaussianBlur:
    """
    Gaussian blur augmentation from SimCLR

    Applies random Gaussian blur with varying kernel sizes and sigma values.
    This is crucial for contrastive learning as it forces the model to learn
    features that are robust to blur.
    """
    def __init__(self, kernel_size: int = 23, sigma: Tuple[float, float] = (0.1, 2.0)):
        """
        Args:
            kernel_size: Size of the Gaussian kernel (must be odd)
            sigma: Range of sigma values for the Gaussian distribution
        """
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class Solarize:
    """
    Solarization augmentation

    Inverts all pixel values above a threshold. This creates unusual
    color patterns that help the model learn more robust features.
    """
    def __init__(self, threshold: int = 128, p: float = 0.5):
        """
        Args:
            threshold: Threshold for solarization (0-255)
            p: Probability of applying solarization
        """
        self.threshold = threshold
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.solarize(img, self.threshold)
        return img


def get_contrastive_augmentation(image_size: int = 224,
                                  grayscale: bool = False,
                                  strength: str = 'strong') -> transforms.Compose:
    """
    Get aggressive augmentation pipeline for contrastive learning

    This follows the SimCLR augmentation strategy:
    1. Random resized crop (provides different views)
    2. Random horizontal flip
    3. Color jitter (very aggressive)
    4. Random grayscale conversion
    5. Gaussian blur
    6. (Optional) Solarization

    Args:
        image_size: Target image size (default 224)
        grayscale: If True, output grayscale images
        strength: Augmentation strength ('weak', 'medium', 'strong')
            - 'weak': Suitable for fine-tuning or when data is limited
            - 'medium': Balanced approach
            - 'strong': Maximum augmentation (SimCLR default)

    Returns:
        Composed transform pipeline
    """

    # Set parameters based on strength
    if strength == 'weak':
        color_jitter_strength = 0.4
        gaussian_blur_p = 0.3
        grayscale_p = 0.1
        crop_scale = (0.7, 1.0)
    elif strength == 'medium':
        color_jitter_strength = 0.6
        gaussian_blur_p = 0.5
        grayscale_p = 0.15
        crop_scale = (0.5, 1.0)
    else:  # 'strong' - SimCLR default
        color_jitter_strength = 0.8
        gaussian_blur_p = 0.5
        grayscale_p = 0.2
        crop_scale = (0.08, 1.0)  # Very aggressive cropping

    # Color jitter parameters (based on color_jitter_strength)
    # SimCLR uses s=1.0, which translates to jitter strength of 0.8
    brightness = 0.8 * color_jitter_strength
    contrast = 0.8 * color_jitter_strength
    saturation = 0.8 * color_jitter_strength
    hue = 0.2 * color_jitter_strength

    # Build augmentation pipeline
    augmentations = [
        transforms.ToPILImage(),
        # Random resized crop - this is KEY for contrastive learning
        # It creates different "views" of the same image
        transforms.RandomResizedCrop(
            size=image_size,
            scale=crop_scale,  # Crop between 8% and 100% of original
            ratio=(0.75, 1.33),  # Aspect ratio range
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    # Color distortions (skip some if grayscale output is needed)
    if not grayscale:
        augmentations.extend([
            # Very aggressive color jitter
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue
                )
            ], p=0.8),
            # Random grayscale conversion
            transforms.RandomGrayscale(p=grayscale_p),
        ])
    else:
        # For grayscale output, still apply brightness/contrast jitter
        augmentations.append(
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast
                )
            ], p=0.8)
        )

    # Gaussian blur - very important for contrastive learning
    augmentations.append(
        transforms.RandomApply([GaussianBlur(kernel_size=23)], p=gaussian_blur_p)
    )

    # Optional: Solarization (uncomment if you want even more aggressive augmentation)
    # augmentations.append(Solarize(threshold=128, p=0.2))

    # Convert to grayscale if needed
    if grayscale:
        augmentations.append(transforms.Grayscale(num_output_channels=1))

    # Convert to tensor and normalize
    augmentations.append(transforms.ToTensor())

    # Normalization
    if grayscale:
        augmentations.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        augmentations.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        )

    return transforms.Compose(augmentations)


def get_simclr_augmentation_pair(image_size: int = 224,
                                   grayscale: bool = False,
                                   strength: str = 'strong') -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get a pair of augmentation pipelines for SimCLR

    Returns two INDEPENDENT augmentation pipelines. When applied to the same image,
    they will create two different augmented views.

    Args:
        image_size: Target image size
        grayscale: If True, output grayscale images
        strength: Augmentation strength ('weak', 'medium', 'strong')

    Returns:
        Tuple of (augmentation1, augmentation2)

    Usage:
        aug1, aug2 = get_simclr_augmentation_pair()
        view1 = aug1(image)  # First augmented view
        view2 = aug2(image)  # Second augmented view (different from view1)
    """
    aug1 = get_contrastive_augmentation(image_size, grayscale, strength)
    aug2 = get_contrastive_augmentation(image_size, grayscale, strength)
    return aug1, aug2


class ContrastiveTransformations:
    """
    Wrapper that applies two different augmentations to create positive pairs

    This is used in the dataset to automatically create augmented pairs.
    """
    def __init__(self, base_transforms: transforms.Compose, n_views: int = 2):
        """
        Args:
            base_transforms: The augmentation pipeline to apply
            n_views: Number of augmented views to create (default 2 for SimCLR)
        """
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        """
        Apply augmentation n_views times to create different views

        Args:
            x: Input image

        Returns:
            List of n_views augmented versions of the input
        """
        return [self.base_transforms(x) for _ in range(self.n_views)]
