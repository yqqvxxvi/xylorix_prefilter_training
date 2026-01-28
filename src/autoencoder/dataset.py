"""
Dataset classes for autoencoder training with feature extraction

This module provides PyTorch Dataset classes that load images and
automatically extract features using the EndGrainFeatureExtractor.

NEW: ImageAutoencoderDataset for deep convolutional autoencoders
     that work directly with raw images instead of handcrafted features.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union
import cv2
from tqdm import tqdm
import pickle
from PIL import Image

from .feature_extractor import EndGrainFeatureExtractor


class FeatureDataset(Dataset):
    """
    PyTorch Dataset that extracts features on-the-fly

    Args:
        image_paths: List of image file paths
        extractor: EndGrainFeatureExtractor instance
        labels: Optional labels for classification (0/1 for endgrain/world)
        cache_features: If True, cache extracted features in memory
        transform: Optional transform to apply to features

    Example:
        >>> extractor = EndGrainFeatureExtractor(normalize=True)
        >>> image_paths = list(Path('data/images').glob('*.jpg'))
        >>> dataset = FeatureDataset(image_paths, extractor)
        >>> features = dataset[0]
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        extractor: EndGrainFeatureExtractor,
        labels: Optional[List[int]] = None,
        cache_features: bool = False,
        transform: Optional[callable] = None
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.extractor = extractor
        self.labels = labels
        self.transform = transform
        self.cache_features = cache_features

        # Validate inputs
        if labels is not None:
            assert len(labels) == len(image_paths), \
                "Number of labels must match number of images"

        # Cache for features
        self.feature_cache = {} if cache_features else None

        # Get feature dimension
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        features, self.feature_names = extractor.extract_all_features(dummy_img)
        self.feature_dim = len(features)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        """
        Get item by index

        Returns:
            features (Tensor) if no labels, else (features, label)
        """
        # Check cache
        if self.feature_cache is not None and idx in self.feature_cache:
            features = self.feature_cache[idx]
        else:
            # Load and extract features
            img_path = self.image_paths[idx]
            features, _ = self.extractor.extract_all_features(str(img_path))

            # Cache if enabled
            if self.feature_cache is not None:
                self.feature_cache[idx] = features

        # Convert to tensor
        features = torch.from_numpy(features).float()

        # Apply transform if provided
        if self.transform is not None:
            features = self.transform(features)

        # Return with or without label
        if self.labels is not None:
            label = self.labels[idx]
            return features, label
        else:
            return features

    def precompute_features(self, verbose: bool = True):
        """
        Precompute and cache all features

        This can significantly speed up training by computing features once.
        """
        if self.feature_cache is None:
            self.feature_cache = {}

        iterator = tqdm(range(len(self)), desc="Extracting features") if verbose else range(len(self))

        for idx in iterator:
            if idx not in self.feature_cache:
                img_path = self.image_paths[idx]
                features, _ = self.extractor.extract_all_features(str(img_path))
                self.feature_cache[idx] = features

    def save_cache(self, cache_path: Union[str, Path]):
        """Save feature cache to disk"""
        if self.feature_cache is None:
            raise ValueError("No cache to save. Enable cache_features=True first.")

        cache_data = {
            'features': self.feature_cache,
            'image_paths': [str(p) for p in self.image_paths],
            'labels': self.labels,
            'feature_names': self.feature_names
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

    @classmethod
    def load_cache(cls, cache_path: Union[str, Path], extractor: EndGrainFeatureExtractor):
        """Load dataset from cached features"""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        dataset = cls(
            image_paths=cache_data['image_paths'],
            extractor=extractor,
            labels=cache_data.get('labels'),
            cache_features=True
        )

        dataset.feature_cache = cache_data['features']
        dataset.feature_names = cache_data['feature_names']

        return dataset


class PrecomputedFeatureDataset(Dataset):
    """
    Dataset for pre-extracted features stored as numpy arrays

    Faster alternative when features are pre-computed and saved.

    Args:
        features: Numpy array of features (N, feature_dim)
        labels: Optional labels (N,)

    Example:
        >>> features = np.random.randn(100, 265)
        >>> dataset = PrecomputedFeatureDataset(features)
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long() if labels is not None else None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Create data loaders for training and validation

    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data

    Returns:
        train_loader or (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, val_loader
    else:
        return train_loader


def load_image_paths_from_directory(
    directory: Union[str, Path],
    extensions: List[str] = ['.jpg', '.jpeg', '.png'],
    recursive: bool = True
) -> List[Path]:
    """
    Load all image paths from a directory

    Args:
        directory: Directory path
        extensions: List of valid image extensions
        recursive: Whether to search recursively

    Returns:
        List of image paths
    """
    directory = Path(directory)
    image_paths = []

    if recursive:
        for ext in extensions:
            image_paths.extend(directory.rglob(f'*{ext}'))
            image_paths.extend(directory.rglob(f'*{ext.upper()}'))
    else:
        for ext in extensions:
            image_paths.extend(directory.glob(f'*{ext}'))
            image_paths.extend(directory.glob(f'*{ext.upper()}'))

    return sorted(image_paths)


def create_endgrain_world_dataset(
    endgrain_dir: Union[str, Path],
    world_dir: Union[str, Path],
    extractor: EndGrainFeatureExtractor,
    val_split: float = 0.2,
    cache_features: bool = True,
    seed: int = 42
) -> Tuple[FeatureDataset, FeatureDataset]:
    """
    Create train/val datasets from endgrain and world directories

    Args:
        endgrain_dir: Directory with end-grain images
        world_dir: Directory with non-wood images
        extractor: Feature extractor instance
        val_split: Validation split ratio (0.0-1.0)
        cache_features: Whether to cache extracted features
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load image paths
    endgrain_paths = load_image_paths_from_directory(endgrain_dir)
    world_paths = load_image_paths_from_directory(world_dir)

    print(f"Found {len(endgrain_paths)} end-grain images")
    print(f"Found {len(world_paths)} world images")

    # Create labels (1 = endgrain, 0 = world)
    endgrain_labels = [1] * len(endgrain_paths)
    world_labels = [0] * len(world_paths)

    # Combine
    all_paths = endgrain_paths + world_paths
    all_labels = endgrain_labels + world_labels

    # Shuffle with seed
    np.random.seed(seed)
    indices = np.random.permutation(len(all_paths))
    all_paths = [all_paths[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    # Split train/val
    n_val = int(len(all_paths) * val_split)
    train_paths = all_paths[n_val:]
    train_labels = all_labels[n_val:]
    val_paths = all_paths[:n_val]
    val_labels = all_labels[:n_val]

    print(f"Train set: {len(train_paths)} images")
    print(f"Val set: {len(val_paths)} images")

    # Create datasets
    train_dataset = FeatureDataset(
        train_paths,
        extractor,
        labels=train_labels,
        cache_features=cache_features
    )

    val_dataset = FeatureDataset(
        val_paths,
        extractor,
        labels=val_labels,
        cache_features=cache_features
    )

    return train_dataset, val_dataset


# ============================================================================
# NEW: Image Dataset for Deep Convolutional Autoencoders
# ============================================================================


class ImageAutoencoderDataset(Dataset):
    """
    Dataset for deep convolutional autoencoders that loads raw images

    This dataset is designed for ONE-CLASS ANOMALY DETECTION:
    - Train on single class (end-grain images only)
    - No labels needed during training (unsupervised)
    - Optionally include labels for validation/testing

    Args:
        image_paths: List of image file paths
        transform: torchvision transforms to apply to images
        labels: Optional labels (0/1 for world/endgrain) for validation/testing
        cache_images: If True, cache loaded images in memory (faster but uses RAM)

    Example (Training):
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ... ])
        >>> image_paths = list(Path('data/endgrain').glob('*.jpg'))
        >>> dataset = ImageAutoencoderDataset(image_paths, transform)
        >>> image = dataset[0]  # Returns single tensor

    Example (Testing with labels):
        >>> dataset = ImageAutoencoderDataset(
        ...     image_paths,
        ...     transform,
        ...     labels=[1, 1, 0, 0, ...]  # 1=endgrain, 0=world
        ... )
        >>> image, label = dataset[0]  # Returns (tensor, label)
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        transform: Optional[callable] = None,
        labels: Optional[List[int]] = None,
        cache_images: bool = False
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform
        self.labels = labels
        self.cache_images = cache_images

        # Validate inputs
        if labels is not None:
            assert len(labels) == len(image_paths), \
                "Number of labels must match number of images"

        # Cache for images
        self.image_cache = {} if cache_images else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self,
        idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        """
        Get item by index

        Returns:
            image (Tensor) if no labels, else (image, label)
        """
        # Check cache
        if self.image_cache is not None and idx in self.image_cache:
            image = self.image_cache[idx]
        else:
            # Load image
            img_path = self.image_paths[idx]
            image = cv2.imread(str(img_path))

            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Cache if enabled
            if self.image_cache is not None:
                self.image_cache[idx] = image

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        # Return with or without label
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image

    def preload_images(self, verbose: bool = True):
        """
        Preload and cache all images in memory

        This speeds up training but uses more RAM.
        """
        if self.image_cache is None:
            self.image_cache = {}

        iterator = tqdm(range(len(self)), desc="Loading images") if verbose else range(len(self))

        for idx in iterator:
            if idx not in self.image_cache:
                img_path = self.image_paths[idx]
                image = cv2.imread(str(img_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.image_cache[idx] = image


def create_oneclass_dataset(
    image_dir: Union[str, Path],
    transform: Optional[callable] = None,
    val_split: float = 0.2,
    seed: int = 42,
    cache_images: bool = False
) -> Tuple[ImageAutoencoderDataset, ImageAutoencoderDataset]:
    """
    Create train/val datasets from a single directory (one-class)

    This is designed for UNSUPERVISED one-class anomaly detection:
    - Loads images from single directory (e.g., end-grain images only)
    - Splits into train/val sets
    - No labels (unsupervised training)

    Args:
        image_dir: Directory with images (single class)
        transform: torchvision transforms to apply
        val_split: Validation split ratio (0.0-1.0)
        seed: Random seed for reproducibility
        cache_images: Whether to cache images in memory

    Returns:
        Tuple of (train_dataset, val_dataset)

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.ToPILImage(),
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ... ])
        >>> train_ds, val_ds = create_oneclass_dataset(
        ...     'data/endgrain',
        ...     transform=transform,
        ...     val_split=0.2
        ... )
    """
    # Load image paths
    image_paths = load_image_paths_from_directory(image_dir)

    print(f"Found {len(image_paths)} images in {image_dir}")

    # Shuffle with seed
    np.random.seed(seed)
    indices = np.random.permutation(len(image_paths))
    image_paths = [image_paths[i] for i in indices]

    # Split train/val
    n_val = int(len(image_paths) * val_split)
    train_paths = image_paths[n_val:]
    val_paths = image_paths[:n_val]

    print(f"Train set: {len(train_paths)} images")
    print(f"Val set: {len(val_paths)} images")

    # Create datasets (no labels for unsupervised training)
    train_dataset = ImageAutoencoderDataset(
        train_paths,
        transform=transform,
        labels=None,
        cache_images=cache_images
    )

    val_dataset = ImageAutoencoderDataset(
        val_paths,
        transform=transform,
        labels=None,
        cache_images=cache_images
    )

    return train_dataset, val_dataset


def create_anomaly_test_dataset(
    endgrain_dir: Union[str, Path],
    world_dir: Union[str, Path],
    transform: Optional[callable] = None,
    cache_images: bool = False
) -> ImageAutoencoderDataset:
    """
    Create test dataset WITH LABELS for anomaly detection evaluation

    This loads both end-grain (inliers) and world (outliers) images
    with labels for computing metrics like accuracy, ROC curve, etc.

    Args:
        endgrain_dir: Directory with end-grain images (inliers)
        world_dir: Directory with world images (outliers)
        transform: torchvision transforms to apply
        cache_images: Whether to cache images in memory

    Returns:
        ImageAutoencoderDataset with labels (1=endgrain, 0=world)

    Example:
        >>> test_ds = create_anomaly_test_dataset(
        ...     'data/endgrain/test',
        ...     'data/world/test',
        ...     transform=transform
        ... )
        >>> image, label = test_ds[0]  # label: 1=endgrain, 0=world
    """
    # Load image paths
    endgrain_paths = load_image_paths_from_directory(endgrain_dir)
    world_paths = load_image_paths_from_directory(world_dir)

    print(f"Found {len(endgrain_paths)} end-grain images")
    print(f"Found {len(world_paths)} world images")

    # Create labels (1 = endgrain/inlier, 0 = world/outlier)
    endgrain_labels = [1] * len(endgrain_paths)
    world_labels = [0] * len(world_paths)

    # Combine
    all_paths = endgrain_paths + world_paths
    all_labels = endgrain_labels + world_labels

    print(f"Total test set: {len(all_paths)} images")

    # Create dataset with labels
    test_dataset = ImageAutoencoderDataset(
        all_paths,
        transform=transform,
        labels=all_labels,
        cache_images=cache_images
    )

    return test_dataset
