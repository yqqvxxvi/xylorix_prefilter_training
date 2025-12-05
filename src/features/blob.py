"""
Blob analysis for wood pore detection and characterization
"""

import cv2
import numpy as np
import math
import itertools
from typing import Dict, Union, Tuple
from pathlib import Path


def euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points

    Args:
        pt1: Point 1 (x, y)
        pt2: Point 2 (x, y)

    Returns:
        Euclidean distance
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def extract_blob_features(
    image: Union[str, Path, np.ndarray],
    threshold_value: int = 40,
    min_blob_area: float = 10,
    max_blob_area: float = 125
) -> Dict[str, float]:
    """
    Extract blob features from wood microscopy image

    This function detects pores (dark regions) in wood images and computes
    geometric features that characterize pore distribution and morphology.

    Args:
        image: Input image (path or BGR numpy array)
        threshold_value: Binary threshold (default 40)
        min_blob_area: Minimum blob area in pixels (default 10)
        max_blob_area: Maximum blob area in pixels (default 125)

    Returns:
        Dictionary with 11 blob features:
            - num_filtered_blobs: Number of detected pores
            - num_centroids: Number of valid centroids
            - avg_blob_area, std_blob_area: Area statistics
            - avg_blob_perimeter, std_blob_perimeter: Perimeter statistics
            - avg_blob_diameter, std_blob_diameter: Equivalent diameter statistics
            - avg_pairwise_centroid_distance: Mean distance between pores
            - std_pairwise_centroid_distance: Std of pairwise distances
            - median_pairwise_centroid_distance: Median pairwise distance

    Example:
        >>> features = extract_blob_features('wood.jpg')
        >>> print(f"Detected {features['num_filtered_blobs']} pores")
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Failed to load image: {image}")
    else:
        img = image

    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)  # Invert: pores are now white

    # Find contours (blobs)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    filtered_contours = [
        cnt for cnt in contours
        if min_blob_area < cv2.contourArea(cnt) < max_blob_area
    ]

    # Extract geometric properties
    centroids = []
    areas = []
    perimeters = []
    diameters = []

    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue

        # Area and perimeter
        areas.append(area)
        perimeters.append(cv2.arcLength(contour, True))

        # Equivalent diameter (circle with same area)
        diameter = math.sqrt(4 * area / math.pi)
        diameters.append(diameter)

        # Centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            centroids.append((cx, cy))

    # Calculate pairwise centroid distances
    pairwise_distances = []
    if len(centroids) >= 2:
        for pt1, pt2 in itertools.combinations(centroids, 2):
            pairwise_distances.append(euclidean_distance(pt1, pt2))

    # Build feature dictionary
    features = {
        'num_filtered_blobs': len(filtered_contours),
        'num_centroids': len(centroids),
        'avg_blob_area': float(np.mean(areas)) if areas else 0.0,
        'std_blob_area': float(np.std(areas)) if areas else 0.0,
        'avg_blob_perimeter': float(np.mean(perimeters)) if perimeters else 0.0,
        'std_blob_perimeter': float(np.std(perimeters)) if perimeters else 0.0,
        'avg_blob_diameter': float(np.mean(diameters)) if diameters else 0.0,
        'std_blob_diameter': float(np.std(diameters)) if diameters else 0.0,
        'avg_pairwise_centroid_distance': float(np.mean(pairwise_distances)) if pairwise_distances else 0.0,
        'std_pairwise_centroid_distance': float(np.std(pairwise_distances)) if pairwise_distances else 0.0,
        'median_pairwise_centroid_distance': float(np.median(pairwise_distances)) if pairwise_distances else 0.0,
    }

    return features


# Standard feature names in order
BLOB_FEATURE_NAMES = [
    'num_filtered_blobs',
    'num_centroids',
    'avg_blob_area',
    'std_blob_area',
    'avg_blob_perimeter',
    'std_blob_perimeter',
    'avg_blob_diameter',
    'std_blob_diameter',
    'avg_pairwise_centroid_distance',
    'std_pairwise_centroid_distance',
    'median_pairwise_centroid_distance'
]
