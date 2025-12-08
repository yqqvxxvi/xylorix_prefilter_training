"""
Data loaders for training and evaluation
"""

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from .dataset import WoodImageDataset, AugmentedWoodDataset
from .transforms import get_train_transforms, get_val_transforms


def get_data_loaders(
    positive_dir: Path,
    negative_dir: Path,
    batch_size: int = 32,
    val_split: float = 0.2,
    image_size: int = 224,
    num_workers: int = 4,
    use_balanced_sampler: bool = False,
    random_seed: int = 42,
    stack_augmentations: bool = False,
    num_augmentations: int = 2,
    include_original: bool = True,
    grayscale: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders

    Args:
        positive_dir: Directory with positive class images (label = 1)
        negative_dir: Directory with negative class images (label = 0)
        batch_size: Batch size
        val_split: Validation split fraction (0.0-1.0)
        image_size: Image resize dimension
        num_workers: Number of data loading workers
        use_balanced_sampler: Use weighted sampler for class imbalance
        random_seed: Random seed for reproducibility
        stack_augmentations: If True, stack augmented images on top of originals
        num_augmentations: Number of augmented versions per image (if stacking)
        include_original: Include original non-augmented images (if stacking)
        grayscale: If True, load images as grayscale (1 channel)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create base dataset (without augmentation if we're going to stack later)
    if stack_augmentations:
        # Use validation transforms for base (no augmentation)
        base_dataset = WoodImageDataset.from_directories(
            positive_dir=positive_dir,
            negative_dir=negative_dir,
            transform=get_val_transforms(image_size, grayscale=grayscale),
            grayscale=grayscale
        )
    else:
        # Use training transforms directly
        base_dataset = WoodImageDataset.from_directories(
            positive_dir=positive_dir,
            negative_dir=negative_dir,
            transform=get_train_transforms(image_size, grayscale=grayscale),
            grayscale=grayscale
        )

    # Split into train and validation
    total_size = len(base_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # Apply augmentation stacking to training set if enabled
    if stack_augmentations:
        # Create a new base dataset for train indices only
        train_base = WoodImageDataset(
            image_paths=[base_dataset.image_paths[i] for i in train_dataset.indices],
            labels=[base_dataset.labels[i] for i in train_dataset.indices],
            transform=get_val_transforms(image_size, grayscale=grayscale),
            grayscale=grayscale
        )

        # Wrap with augmented dataset
        train_dataset = AugmentedWoodDataset(
            base_dataset=train_base,
            augmentation_transform=get_train_transforms(image_size, grayscale=grayscale),
            num_augmentations=num_augmentations,
            include_original=include_original,
            grayscale=grayscale
        )

        multiplier = (1 if include_original else 0) + num_augmentations
        print(f"Augmentation stacking enabled: {num_augmentations} augmented versions per image")
        print(f"Training set multiplied by {multiplier}x: {len(train_base)} -> {len(train_dataset)} samples")

    # Ensure validation uses val transforms
    if not stack_augmentations:
        val_dataset.dataset.transform = get_val_transforms(image_size, grayscale=grayscale)

    # Create sampler for class imbalance (optional)
    sampler = None
    if use_balanced_sampler:
        if stack_augmentations:
            # For stacked augmentations, get all labels from the augmented dataset
            # Each original image is repeated (multiplier) times
            base_labels = train_dataset.base_dataset.labels
            multiplier = (1 if include_original else 0) + num_augmentations
            train_labels = []
            for label in base_labels:
                train_labels.extend([label] * multiplier)
        else:
            # Get labels for training set normally
            train_labels = [base_dataset.labels[i] for i in train_dataset.indices]

        # Calculate class weights
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        print(f"Using balanced sampler. Class counts: {class_counts}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),  # Don't shuffle if using sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    if not stack_augmentations:
        print(f"Train set: {train_size} samples")
    print(f"Val set: {val_size} samples")

    return train_loader, val_loader


def get_test_loader(
    test_dir: Optional[Path] = None,
    positive_dir: Optional[Path] = None,
    negative_dir: Optional[Path] = None,
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 4,
    grayscale: bool = False
) -> DataLoader:
    """
    Create test data loader

    Args:
        test_dir: Directory with test images (if organized as positive/negative subdirs)
        positive_dir: Directory with positive class test images
        negative_dir: Directory with negative class test images
        batch_size: Batch size
        image_size: Image resize dimension
        num_workers: Number of data loading workers
        grayscale: If True, load images as grayscale (1 channel)

    Returns:
        Test data loader
    """
    if test_dir is not None:
        # Assume test_dir has positive/ and negative/ subdirectories
        positive_test = test_dir / 'positive'
        negative_test = test_dir / 'negative'
    else:
        positive_test = positive_dir
        negative_test = negative_dir

    test_dataset = WoodImageDataset.from_directories(
        positive_dir=positive_test,
        negative_dir=negative_test,
        transform=get_val_transforms(image_size, grayscale=grayscale),
        grayscale=grayscale
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Test set: {len(test_dataset)} samples")

    return test_loader
