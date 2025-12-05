"""
Dataset classes for wood classification
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import pandas as pd


class WoodImageDataset(Dataset):
    """
    PyTorch Dataset for binary image classification

    Supports two modes:
    1. Directory-based: positive_dir/ and negative_dir/ with images
    2. CSV-based: CSV file with image paths and labels
    """

    def __init__(self,
                 image_paths: List[Path],
                 labels: List[int],
                 transform: Optional[Callable] = None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (0 = non-wood, 1 = wood)
            transform: Optional transform to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        assert len(self.image_paths) == len(self.labels), \
            "Number of images and labels must match"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image (BGR)
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @classmethod
    def from_directories(cls,
                        positive_dir: Path,
                        negative_dir: Path,
                        transform: Optional[Callable] = None):
        """
        Create dataset from two directories

        Args:
            positive_dir: Directory containing positive class images (label = 1)
            negative_dir: Directory containing negative class images (label = 0)
            transform: Optional transform to apply

        Returns:
            WoodImageDataset instance
        """
        positive_dir = Path(positive_dir)
        negative_dir = Path(negative_dir)

        # Collect positive class images (label = 1)
        positive_paths = list(positive_dir.glob('*.jpg')) + \
                        list(positive_dir.glob('*.jpeg')) + \
                        list(positive_dir.glob('*.png'))
        positive_labels = [1] * len(positive_paths)

        # Collect negative class images (label = 0)
        negative_paths = list(negative_dir.glob('*.jpg')) + \
                        list(negative_dir.glob('*.jpeg')) + \
                        list(negative_dir.glob('*.png'))
        negative_labels = [0] * len(negative_paths)

        # Combine
        all_paths = positive_paths + negative_paths
        all_labels = positive_labels + negative_labels

        print(f"Loaded {len(positive_paths)} positive images, {len(negative_paths)} negative images")

        return cls(all_paths, all_labels, transform=transform)

    @classmethod
    def from_csv(cls,
                 csv_path: Path,
                 image_col: str = 'image_path',
                 label_col: str = 'label',
                 transform: Optional[Callable] = None):
        """
        Create dataset from CSV file

        Args:
            csv_path: Path to CSV file
            image_col: Column name for image paths
            label_col: Column name for labels
            transform: Optional transform to apply

        Returns:
            WoodImageDataset instance

        CSV format:
            image_path,label
            /path/to/image1.jpg,1
            /path/to/image2.jpg,0
        """
        df = pd.read_csv(csv_path)

        image_paths = [Path(p) for p in df[image_col].values]
        labels = df[label_col].values.tolist()

        print(f"Loaded {len(image_paths)} images from {csv_path}")

        return cls(image_paths, labels, transform=transform)


class AugmentedWoodDataset(Dataset):
    """
    Wrapper dataset that stacks augmented versions on top of originals
    to increase dataset size for training.

    If num_augmentations=2, each original image will have 2 augmented
    versions, tripling the effective dataset size.
    """

    def __init__(self,
                 base_dataset: WoodImageDataset,
                 augmentation_transform: Callable,
                 num_augmentations: int = 1,
                 include_original: bool = True):
        """
        Args:
            base_dataset: Base WoodImageDataset (should use val transforms, not train)
            augmentation_transform: Transform to apply for augmentations
            num_augmentations: Number of augmented versions per original image
            include_original: Whether to include original (non-augmented) images
        """
        self.base_dataset = base_dataset
        self.augmentation_transform = augmentation_transform
        self.num_augmentations = num_augmentations
        self.include_original = include_original

        # Calculate multiplier: original + augmented versions
        self.multiplier = (1 if include_original else 0) + num_augmentations

    def __len__(self) -> int:
        return len(self.base_dataset) * self.multiplier

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label

        Returns:
            Tuple of (image_tensor, label)
        """
        # Determine which original image this corresponds to
        original_idx = idx // self.multiplier
        augmentation_idx = idx % self.multiplier

        # Get the original image (raw RGB numpy array)
        img_path = self.base_dataset.image_paths[original_idx]
        label = self.base_dataset.labels[original_idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # If we want the original and this is the first index
        if self.include_original and augmentation_idx == 0:
            # Apply base transform (without augmentation)
            if self.base_dataset.transform is not None:
                image = self.base_dataset.transform(image)
        else:
            # Apply augmentation transform
            image = self.augmentation_transform(image)

        return image, label


class WoodFeatureDataset(Dataset):
    """
    Dataset for pre-extracted features (for Random Forest, MLP)

    Loads features from CSV or numpy arrays
    """

    def __init__(self,
                 features: np.ndarray,
                 labels: np.ndarray):
        """
        Args:
            features: Feature array (N, D)
            labels: Label array (N,)
        """
        self.features = features
        self.labels = labels

        assert len(self.features) == len(self.labels), \
            "Number of features and labels must match"

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get features and label"""
        return self.features[idx], self.labels[idx]

    @classmethod
    def from_csv(cls,
                 csv_path: Path,
                 feature_cols: Optional[List[str]] = None,
                 label_col: str = 'label'):
        """
        Load features from CSV file

        Args:
            csv_path: Path to CSV file
            feature_cols: List of feature column names (None = all except label)
            label_col: Label column name

        Returns:
            WoodFeatureDataset instance
        """
        df = pd.read_csv(csv_path)

        if feature_cols is None:
            # Use all columns except label column
            feature_cols = [col for col in df.columns if col != label_col]

        features = df[feature_cols].values
        labels = df[label_col].values

        print(f"Loaded {len(features)} samples with {features.shape[1]} features")

        return cls(features, labels)
