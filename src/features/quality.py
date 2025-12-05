"""
Variance of Laplacian (VoL) for image quality/blur detection
"""

import cv2
import numpy as np
from typing import Union
from pathlib import Path


def calculate_vol(image: Union[str, Path, np.ndarray], crop: bool = True) -> float:
    """
    Calculate Variance of Laplacian (VoL) for sharpness detection

    Args:
        image: Image path or numpy array (BGR format)
        crop: Whether to crop to center region (320x320)

    Returns:
        VoL score (higher = sharper image)

    Example:
        >>> vol_score = calculate_vol('image.jpg')
        >>> is_sharp = vol_score >= 900
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Failed to load image: {image}")
    else:
        img = image

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply center crop if requested
    if crop:
        gray = gray[140:460, 40:360]  # Center 320x320 region

    # Calculate Laplacian and variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    vol_score = float(laplacian.var())

    return vol_score


def is_image_clear(image: Union[str, Path, np.ndarray],
                   threshold: float = 900,
                   crop: bool = True) -> bool:
    """
    Check if image is clear (not blurry)

    Args:
        image: Image path or numpy array
        threshold: VoL threshold (default 900)
        crop: Whether to crop to center region

    Returns:
        True if image is clear, False if blurry
    """
    vol_score = calculate_vol(image, crop=crop)
    return vol_score >= threshold
