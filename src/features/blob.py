"""
Blob analysis for wood pore detection and characterization

This module implements a sequential region-finding algorithm for blob detection
without using OpenCV's findContours. It uses a two-pass connected component
labeling algorithm with Union-Find for equivalence tracking.
"""

from PIL import Image
import numpy as np
import math
import itertools
from typing import Dict, Union, Tuple, Set
from pathlib import Path


class UnionFind:
    """
    Union-Find data structure for tracking region label equivalences

    Uses path compression and union-by-rank for optimal performance.
    """

    def __init__(self, n: int):
        """
        Initialize Union-Find structure

        Args:
            n: Maximum number of labels
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """
        Find root of element x with path compression

        Args:
            x: Element to find root of

        Returns:
            Root element of x's equivalence class
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """
        Merge equivalence classes of x and y using union-by-rank

        Args:
            x: First element
            y: Second element
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank: attach smaller tree under larger tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load image using PIL and convert to BGR numpy array

    Args:
        image_path: Path to image file

    Returns:
        numpy array in BGR format (H, W, 3)

    Raises:
        ValueError: If image cannot be loaded
    """
    try:
        pil_img = Image.open(str(image_path))
        rgb_array = np.array(pil_img)

        # Handle grayscale images
        if len(rgb_array.shape) == 2:
            # Convert grayscale to BGR
            bgr_array = np.stack([rgb_array, rgb_array, rgb_array], axis=2)
        else:
            # Convert RGB to BGR (to match cv2.imread behavior)
            bgr_array = rgb_array[:, :, ::-1].copy()

        return bgr_array
    except Exception as e:
        raise ValueError(f"Failed to load image: {image_path}") from e


def bgr_to_gray(bgr_image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale

    Uses standard luminosity method: 0.299*R + 0.587*G + 0.114*B

    Args:
        bgr_image: numpy array (H, W, 3) in BGR format

    Returns:
        Grayscale image (H, W) as uint8
    """
    # Handle already grayscale images
    if len(bgr_image.shape) == 2:
        return bgr_image

    if bgr_image.shape[2] == 1:
        return bgr_image[:, :, 0]

    # Extract BGR channels
    B = bgr_image[:, :, 0].astype(np.float32)
    G = bgr_image[:, :, 1].astype(np.float32)
    R = bgr_image[:, :, 2].astype(np.float32)

    # Apply luminosity formula
    gray = 0.299 * R + 0.587 * G + 0.114 * B

    # Convert back to uint8
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def threshold_binary(gray_image: np.ndarray, threshold_value: int) -> np.ndarray:
    """
    Apply binary threshold to grayscale image

    Args:
        gray_image: Grayscale image (H, W)
        threshold_value: Threshold value

    Returns:
        Binary image: pixels > threshold = 255, else 0
    """
    binary = np.where(gray_image > threshold_value, 255, 0).astype(np.uint8)
    return binary


def preprocess_image(
    image: Union[str, Path, np.ndarray],
    threshold_value: int,
    resize_to: tuple = (320, 320)
) -> np.ndarray:
    """
    Preprocess image: load -> resize -> grayscale -> threshold -> invert -> normalize

    Args:
        image: Input image (path or BGR numpy array)
        threshold_value: Binary threshold
        resize_to: Target size (width, height) for resizing. Default (320, 320)

    Returns:
        Binary mask (0 = background, 1 = foreground/pores)
    """
    # Load if path
    if isinstance(image, (str, Path)):
        img = load_image(str(image))
    else:
        img = image

    # Resize image to target size
    if resize_to is not None:
        from PIL import Image as PILImage
        h, w = img.shape[:2]
        if (w, h) != resize_to:
            # Convert to PIL for resizing
            img_pil = PILImage.fromarray(img[:, :, ::-1])  # BGR to RGB
            img_pil = img_pil.resize(resize_to, PILImage.LANCZOS)
            img = np.array(img_pil)[:, :, ::-1]  # RGB back to BGR

    # Convert to grayscale
    gray = bgr_to_gray(img)

    # Apply threshold
    binary = threshold_binary(gray, threshold_value)

    # Invert (pores are now 255, background is 0)
    binary = 255 - binary

    # Normalize to 0/1 for easier processing
    binary_mask = (binary > 0).astype(np.int32)

    return binary_mask


def connected_components_labeling(binary_mask: np.ndarray) -> np.ndarray:
    """
    Two-pass connected component labeling with sequential region finding

    Uses a 2x2 sliding window with specific labeling rules and Union-Find
    for equivalence tracking.

    Window layout:
        C(row-1, col-1)    D(row-1, col)
        B(row, col-1)      A(row, col)    <- Current pixel

    Args:
        binary_mask: Binary image (0 = background, 1 = foreground)

    Returns:
        Label matrix where each connected region has a unique label
    """
    height, width = binary_mask.shape
    labels = np.zeros((height, width), dtype=np.int32)

    # Initialize Union-Find with upper bound on number of labels
    max_labels = height * width // 2 + 1000
    uf = UnionFind(max_labels)

    next_label = 1

    # Pass 1: Scan and assign initial labels
    for row in range(height):
        for col in range(width):
            # Rule 1: If A is background, skip it
            if binary_mask[row, col] == 0:
                labels[row, col] = 0
                continue

            # Get neighbor labels (treat out-of-bounds as 0)
            label_B = labels[row, col - 1] if col > 0 else 0
            label_C = labels[row - 1, col - 1] if row > 0 and col > 0 else 0
            label_D = labels[row - 1, col] if row > 0 else 0

            # Rule 2: If C is labeled region n, assign A as n
            if label_C > 0:
                labels[row, col] = label_C

                # Also check for equivalences with B and D
                if label_B > 0 and label_B != label_C:
                    uf.union(label_C, label_B)
                if label_D > 0 and label_D != label_C:
                    uf.union(label_C, label_D)

            # Rule 3: If D is background and B is labeled n, assign A as n
            elif label_D == 0 and label_B > 0:
                labels[row, col] = label_B

            # Rule 4: If B is background and D is labeled n, assign A as n
            elif label_B == 0 and label_D > 0:
                labels[row, col] = label_D

            # Rule 5: If B and D are both labeled n, assign A as n
            elif label_B > 0 and label_D > 0 and label_B == label_D:
                labels[row, col] = label_B

            # Rule 6: If B is labeled n and D is labeled m, assign A as n and note m≡n
            elif label_B > 0 and label_D > 0 and label_B != label_D:
                labels[row, col] = label_B
                uf.union(label_B, label_D)

            # Rule 7: If B, C, D are all background, assign new label
            elif label_B == 0 and label_C == 0 and label_D == 0:
                labels[row, col] = next_label
                next_label += 1

    # Pass 2: Consolidate labels using Union-Find
    for row in range(height):
        for col in range(width):
            if labels[row, col] > 0:
                labels[row, col] = uf.find(labels[row, col])

    return labels


def calculate_perimeter(
    pixel_coords: Set[Tuple[int, int]],
    height: int,
    width: int
) -> int:
    """
    Calculate perimeter by counting boundary pixels

    A pixel is a boundary pixel if at least one of its 8-connected neighbors
    is outside the region or out of bounds.

    Args:
        pixel_coords: Set of (row, col) coordinates in the region
        height: Image height
        width: Image width

    Returns:
        Perimeter count
    """
    perimeter = 0
    neighbors_8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for row, col in pixel_coords:
        is_boundary = False
        for dr, dc in neighbors_8:
            nr, nc = row + dr, col + dc

            # Boundary if neighbor is out of bounds or outside region
            if (nr < 0 or nr >= height or nc < 0 or nc >= width or
                (nr, nc) not in pixel_coords):
                is_boundary = True
                break

        if is_boundary:
            perimeter += 1

    return perimeter


def extract_region_properties(labels: np.ndarray) -> Dict[int, Dict]:
    """
    Extract properties from labeled regions

    Calculates moments, area, perimeter, centroid, and diameter for each region.

    Args:
        labels: Label matrix from connected_components_labeling

    Returns:
        Dictionary mapping label to properties:
            - m00: Zeroth moment (area)
            - m10, m01: First moments
            - area: Region area in pixels
            - perimeter: Perimeter in pixels
            - centroid: (cx, cy) tuple
            - diameter: Equivalent diameter
    """
    height, width = labels.shape
    regions = {}

    # Single pass: accumulate moments and collect pixels
    for row in range(height):
        for col in range(width):
            label = labels[row, col]
            if label == 0:  # Skip background
                continue

            if label not in regions:
                regions[label] = {
                    'm00': 0,
                    'm10': 0,
                    'm01': 0,
                    'pixels': set()
                }

            # Accumulate moments
            regions[label]['m00'] += 1        # Area = count of pixels
            regions[label]['m10'] += col      # Sum of x coordinates
            regions[label]['m01'] += row      # Sum of y coordinates
            regions[label]['pixels'].add((row, col))

    # Calculate derived properties
    for label, data in regions.items():
        m00 = data['m00']

        if m00 == 0:  # Safety check
            continue

        # Area
        area = m00
        data['area'] = area

        # Centroid (center of mass)
        centroid_x = float(data['m10'] / m00)
        centroid_y = float(data['m01'] / m00)
        data['centroid'] = (centroid_x, centroid_y)

        # Equivalent diameter (circle with same area)
        diameter = math.sqrt(4 * area / math.pi)
        data['diameter'] = diameter

        # Perimeter
        perimeter = calculate_perimeter(data['pixels'], height, width)
        data['perimeter'] = perimeter

    return regions


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
    threshold_value: int = 80,
    min_blob_area: float = 3,
    max_blob_area: float = 40,
    resize_to: tuple = (320, 320)
) -> Dict[str, float]:
    """
    Extract blob features from wood microscopy image

    This function detects pores (dark regions) in wood images and computes
    geometric features that characterize pore distribution and morphology.

    Uses a sequential region-finding algorithm with two-pass connected
    component labeling instead of OpenCV's findContours.

    Args:
        image: Input image (path or BGR numpy array)
        threshold_value: Binary threshold (default 40)
        min_blob_area: Minimum blob area in pixels (default 3)
        max_blob_area: Maximum blob area in pixels (default 40)
        resize_to: Resize image to (width, height) before processing (default 320x320)

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
    # Step 1: Preprocess image (resize to 320x320, then threshold)
    binary_mask = preprocess_image(image, threshold_value, resize_to=resize_to)

    # Step 2: Connected component labeling
    labels = connected_components_labeling(binary_mask)

    # Step 3: Extract region properties
    regions = extract_region_properties(labels)

    # Step 4: Filter by area
    filtered_regions = {
        label: props
        for label, props in regions.items()
        if min_blob_area < props['area'] < max_blob_area
    }

    # Handle empty case
    if len(filtered_regions) == 0:
        return {
            'num_filtered_blobs': 0,
            'num_centroids': 0,
            'avg_blob_area': 0.0,
            'std_blob_area': 0.0,
            'avg_blob_perimeter': 0.0,
            'std_blob_perimeter': 0.0,
            'avg_blob_diameter': 0.0,
            'std_blob_diameter': 0.0,
            'avg_pairwise_centroid_distance': 0.0,
            'std_pairwise_centroid_distance': 0.0,
            'median_pairwise_centroid_distance': 0.0,
        }

    # Extract lists for statistics
    areas = [props['area'] for props in filtered_regions.values()]
    perimeters = [props['perimeter'] for props in filtered_regions.values()]
    diameters = [props['diameter'] for props in filtered_regions.values()]
    centroids = [props['centroid'] for props in filtered_regions.values()]

    # Calculate pairwise centroid distances
    pairwise_distances = []
    if len(centroids) >= 2:
        for pt1, pt2 in itertools.combinations(centroids, 2):
            pairwise_distances.append(euclidean_distance(pt1, pt2))

    # Build feature dictionary
    features = {
        'num_filtered_blobs': len(filtered_regions),
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
