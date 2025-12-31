"""
Contrastive learning loss functions

Implements NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
and other contrastive losses for self-supervised learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)

    This is the loss function used in SimCLR and many other contrastive learning methods.
    Also known as InfoNCE loss.

    The loss pulls together representations of positive pairs (augmentations of the same image)
    while pushing apart representations of negative pairs (different images).

    Formula:
        For a positive pair (i, j), the loss is:
        l(i,j) = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]

        where:
        - sim(u,v) = u^T v / (||u|| ||v||)  [cosine similarity]
        - τ is the temperature parameter
        - k iterates over all negative samples in the batch

    Key Properties:
    1. Temperature (τ): Controls the concentration of the distribution
       - Lower τ → sharper distribution, focuses on hardest negatives
       - Higher τ → smoother distribution, considers all negatives more equally
       - Typical values: 0.1 - 0.5

    2. Batch Size: Larger batches provide more negative samples, leading to better performance
       - SimCLR paper used batch sizes of 4096-8192
       - For smaller GPUs, use gradient accumulation or smaller batches (256-512)

    3. Similarity Metric: Uses cosine similarity (normalized dot product)
    """

    def __init__(self, temperature: float = 0.5, use_cosine_similarity: bool = True):
        """
        Args:
            temperature: Temperature parameter τ for scaling (default 0.5)
                - Lower values (0.1-0.3): More aggressive, focuses on hard negatives
                - Medium values (0.4-0.6): Balanced approach
                - Higher values (0.7-1.0): Softer, considers all negatives equally
            use_cosine_similarity: If True, use cosine similarity; else use dot product
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for a batch of positive pairs

        Args:
            z_i: Embeddings of first augmented view (batch_size, embedding_dim)
            z_j: Embeddings of second augmented view (batch_size, embedding_dim)
                 z_i[k] and z_j[k] are positive pairs (same original image)

        Returns:
            Scalar loss value

        How it works:
        1. For each sample i, its positive pair is j (same image, different augmentation)
        2. All other samples in the batch are negative pairs
        3. The loss encourages high similarity between (i,j) and low similarity with negatives
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings (required for cosine similarity)
        if self.use_cosine_similarity:
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

        # Concatenate z_i and z_j to form full batch
        # Shape: (2 * batch_size, embedding_dim)
        representations = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        # Shape: (2 * batch_size, 2 * batch_size)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        ) if self.use_cosine_similarity else torch.mm(representations, representations.T)

        # Create mask to identify positive pairs
        # For batch_size=4, the positive pairs are:
        # (0,4), (1,5), (2,6), (3,7) and their symmetric pairs
        mask = torch.eye(batch_size, dtype=torch.bool, device=z_i.device)
        # Expand mask to full 2N x 2N matrix
        positives_mask = mask.repeat(2, 2)

        # Create mask for negatives (all pairs except self and positive)
        # Exclude diagonal (self-similarity)
        negatives_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)

        # Extract positive pairs
        # For each sample i in first half, positive is i+batch_size in second half
        # For each sample j in second half, positive is j-batch_size in first half
        positives = similarity_matrix[positives_mask].view(2 * batch_size, -1)

        # Extract negative pairs
        negatives = similarity_matrix[negatives_mask].view(2 * batch_size, -1)

        # Compute logits with temperature scaling
        logits = torch.cat([positives, negatives], dim=1) / self.temperature

        # Labels: positive pair is always the first column (index 0)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels, reduction='mean')

        return loss


class SimCLRLoss(nn.Module):
    """
    SimCLR loss - alternative implementation with clearer structure

    This implementation makes the anchor-positive-negative relationship more explicit.
    Functionally equivalent to NTXentLoss but easier to understand.
    """

    def __init__(self, temperature: float = 0.5):
        """
        Args:
            temperature: Temperature parameter for scaling
        """
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute SimCLR loss

        Args:
            z_i: First view embeddings (batch_size, embedding_dim)
            z_j: Second view embeddings (batch_size, embedding_dim)

        Returns:
            Loss value
        """
        batch_size = z_i.shape[0]

        # L2 normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate both views
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, dim)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.T) / self.temperature  # (2*batch_size, 2*batch_size)

        # Create positive pair indices
        # For i in [0, batch_size), positive is i + batch_size
        # For i in [batch_size, 2*batch_size), positive is i - batch_size
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)

        # Create mask to remove self-similarity (diagonal)
        mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        mask.fill_diagonal_(False)

        # For each sample, compute loss
        # We want the similarity to positive to be highest among all pairs
        total_loss = 0
        for i in range(2 * batch_size):
            # Get positive pair similarity
            pos_sim = sim_matrix[i, pos_indices[i]]

            # Get all similarities except self
            neg_sim = sim_matrix[i][mask[i]]

            # Combine: positive should have highest similarity
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])

            # Target: positive (index 0) should be chosen
            labels = torch.zeros(1, dtype=torch.long, device=z.device)

            # Compute cross-entropy loss
            total_loss += self.criterion(logits.unsqueeze(0), labels)

        return total_loss / (2 * batch_size)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon)

    Extension of NT-Xent that uses label information.
    When labels are available, samples with the same label are treated as positives.

    This can be useful for:
    1. Semi-supervised learning (mix labeled and unlabeled data)
    2. Comparing self-supervised vs supervised contrastive learning
    3. Fine-tuning with limited labels

    Reference: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
    """

    def __init__(self, temperature: float = 0.5):
        """
        Args:
            temperature: Temperature parameter
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss

        Args:
            features: Embeddings (batch_size * n_views, embedding_dim)
                      For 2 views, this should be [view1; view2] concatenated
            labels: Class labels (batch_size,)

        Returns:
            Loss value
        """
        device = features.device
        batch_size = labels.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(features, features.T) / self.temperature

        # Create label mask: 1 if same label, 0 otherwise
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Remove diagonal (self-similarity)
        logits_mask = torch.ones_like(mask).scatter_(1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device), 0)

        # Mask out self-similarity
        mask = mask * logits_mask

        # Compute log probability
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = -mean_log_prob_pos.mean()

        return loss
