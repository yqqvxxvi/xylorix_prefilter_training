"""
Image transformations for data augmentation
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple
import random
from PIL import Image


class ResizeAndNormalize:
    """Resize image and normalize to [0, 1]"""

    def __init__(self, size: Tuple[int, int] = (224, 224)):
        self.size = size

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Args:
            image: RGB numpy array (H, W, 3)

        Returns:
            Tensor (3, H, W) normalized to [0, 1]
        """
        # Resize
        resized = cv2.resize(image, self.size)

        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        return tensor


class RandomRotationReflect:
    """
    Random rotation with reflected/mirrored borders instead of black edges.

    Uses OpenCV's rotation with BORDER_REFLECT_101 to fill edges with
    mirrored content instead of black pixels.
    """

    def __init__(self, degrees: float = 180.0):
        """
        Args:
            degrees: Maximum rotation angle in both directions (0 to ±degrees)
        """
        self.degrees = degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img: PIL Image

        Returns:
            Rotated PIL Image with reflected borders
        """
        # Random angle between -degrees and +degrees
        angle = random.uniform(-self.degrees, self.degrees)

        # Convert PIL to numpy array
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        # Get rotation matrix
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

        # Rotate with reflected border
        rotated = cv2.warpAffine(
            img_array,
            rotation_matrix,
            (w, h),
            borderMode=cv2.BORDER_REFLECT_101  # Mirror the edges
        )

        # Convert back to PIL
        return Image.fromarray(rotated)


def get_train_transforms(image_size: int = 224):
    """
    Get training transforms with data augmentation

    Args:
        image_size: Target image size

    Returns:
        Transform function
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomRotationReflect(degrees=180),  # Rotation with mirrored edges
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size: int = 224):
    """
    Get validation transforms (no augmentation)

    Args:
        image_size: Target image size

    Returns:
        Transform function
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_test_transforms(image_size: int = 224):
    """
    Get test transforms (same as validation)

    Args:
        image_size: Target image size

    Returns:
        Transform function
    """
    return get_val_transforms(image_size)


def get_inference_transforms(image_size: int = 224):
    """
    Get inference transforms (same as validation/test)

    Args:
        image_size: Target image size

    Returns:
        Transform function
    """
    return get_val_transforms(image_size)
