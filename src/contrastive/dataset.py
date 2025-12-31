"""
Dataset wrapper for contrastive learning

Wraps existing datasets to provide multiple augmented views of each image.
"""

import torch
from torch.utils.data import Dataset
from typing import Callable, Optional
import numpy as np


class ContrastiveDataset(Dataset):
    """
    Wraps a standard dataset to provide multiple augmented views

    For contrastive learning, we need multiple augmented versions of each image.
    This wrapper applies the same augmentation pipeline multiple times to create
    different "views" of each sample.

    Example:
        Original image → [augmentation] → view1
                      → [augmentation] → view2

        view1 and view2 are the positive pair (anchor and positive)
        Other images in the batch are negatives
    """

    def __init__(self,
                 base_dataset: Dataset,
                 transform: Callable,
                 n_views: int = 2,
                 return_label: bool = False):
        """
        Args:
            base_dataset: The underlying dataset (should return (image, label))
            transform: Augmentation transform to apply
                       Should be ContrastiveTransformations or similar that returns a list
            n_views: Number of augmented views to create (default 2)
            return_label: If True, also return the label (useful for supervised contrastive)
        """
        self.base_dataset = base_dataset
        self.transform = transform
        self.n_views = n_views
        self.return_label = return_label

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        """
        Get item with multiple augmented views

        Returns:
            If return_label=False:
                views: List of n_views augmented tensors
            If return_label=True:
                (views, label): Tuple of (list of views, label)
        """
        # Get original image and label from base dataset
        image, label = self.base_dataset[idx]

        # Apply transform to create multiple views
        # The transform should be ContrastiveTransformations which returns a list
        views = self.transform(image)

        if self.return_label:
            return views, label
        else:
            return views


class ContrastiveDatasetWrapper(Dataset):
    """
    Alternative implementation: applies transform multiple times manually

    Use this if you have a standard transform (not ContrastiveTransformations)
    """

    def __init__(self,
                 base_dataset: Dataset,
                 transform: Callable,
                 n_views: int = 2,
                 return_label: bool = False):
        """
        Args:
            base_dataset: The underlying dataset
            transform: Standard transform (will be applied n_views times)
            n_views: Number of views to create
            return_label: Whether to return labels
        """
        self.base_dataset = base_dataset
        self.transform = transform
        self.n_views = n_views
        self.return_label = return_label

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        """Get item with multiple augmented views"""
        image, label = self.base_dataset[idx]

        # Apply transform n_views times to create different augmentations
        views = [self.transform(image) for _ in range(self.n_views)]

        if self.return_label:
            return views, label
        else:
            return views


def contrastive_collate_fn(batch):
    """
    Custom collate function for contrastive learning

    Converts a batch of [view1, view2] lists into separate tensors.

    Args:
        batch: List of samples, where each sample is a list of views

    Returns:
        If batch contains (views, labels):
            Tuple of (view_tensors, labels)
        If batch contains only views:
            view_tensors

    Example:
        Input batch (size=4, n_views=2):
            [ [view1_img0, view2_img0],
              [view1_img1, view2_img1],
              [view1_img2, view2_img2],
              [view1_img3, view2_img3] ]

        Output:
            [tensor([view1_img0, view1_img1, view1_img2, view1_img3]),
             tensor([view2_img0, view2_img1, view2_img2, view2_img3])]
    """
    # Check if batch contains labels
    if isinstance(batch[0], tuple):
        # Batch has (views, label) tuples
        views_list = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])

        # Stack views
        n_views = len(views_list[0])
        view_tensors = [torch.stack([views[i] for views in views_list])
                        for i in range(n_views)]

        return view_tensors, labels
    else:
        # Batch has only views (no labels)
        # Each item is a list of n_views tensors
        n_views = len(batch[0])

        # Stack each view across the batch
        # view_tensors[i] contains the i-th view of all samples in the batch
        view_tensors = [torch.stack([item[i] for item in batch])
                        for i in range(n_views)]

        return view_tensors
