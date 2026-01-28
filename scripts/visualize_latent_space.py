"""
Visualize latent space of trained autoencoder

This script loads a trained autoencoder and visualizes the latent space
embeddings using dimensionality reduction techniques (t-SNE, PCA, UMAP).

Usage:
    python scripts/visualize_latent_space.py \
        --checkpoint results/autoencoder/checkpoints/best_model.pth \
        --endgrain_dir path/to/endgrain \
        --world_dir path/to/world \
        --output_dir results/latent_viz
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.autoencoder import (
    EndGrainFeatureExtractor,
    FeatureAutoencoder,
    VariationalAutoencoder,
    FeatureDataset,
    load_image_paths_from_directory
)


def extract_latent_representations(
    model,
    dataset,
    device='cuda',
    batch_size=32
):
    """
    Extract latent representations for all samples in dataset

    Args:
        model: Trained autoencoder model
        dataset: Feature dataset
        device: Device to use
        batch_size: Batch size

    Returns:
        Tuple of (latent_vectors, labels) if dataset has labels, else latent_vectors
    """
    model.eval()
    model = model.to(device)

    latent_vectors = []
    labels_list = []
    has_labels = hasattr(dataset, 'labels') and dataset.labels is not None

    # Create data loader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting latent representations"):
            if isinstance(batch, tuple):
                features, labels = batch
                labels_list.extend(labels.cpu().numpy())
            else:
                features = batch

            features = features.to(device)

            # Get latent representation
            if hasattr(model, 'encode'):
                if isinstance(model, VariationalAutoencoder):
                    mu, logvar = model.encode(features)
                    latent = mu  # Use mean for visualization
                else:
                    latent = model.encode(features)
            else:
                _, latent = model(features)

            latent_vectors.append(latent.cpu().numpy())

    latent_vectors = np.vstack(latent_vectors)

    if has_labels:
        labels_array = np.array(labels_list)
        return latent_vectors, labels_array
    else:
        return latent_vectors


def visualize_latent_space_2d(
    latent_vectors,
    labels=None,
    method='pca',
    output_path=None
):
    """
    Visualize latent space in 2D using dimensionality reduction

    Args:
        latent_vectors: Latent representations (N, latent_dim)
        labels: Optional labels (N,)
        method: 'pca', 'tsne', or 'umap'
        output_path: Path to save plot
    """
    print(f"\nReducing to 2D using {method.upper()}...")

    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        embeddings = reducer.fit_transform(latent_vectors)
        explained_var = reducer.explained_variance_ratio_
        print(f"Explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}")

    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings = reducer.fit_transform(latent_vectors)

    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings = reducer.fit_transform(latent_vectors)
        except ImportError:
            print("UMAP not installed. Install with: pip install umap-learn")
            print("Falling back to t-SNE...")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embeddings = reducer.fit_transform(latent_vectors)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Create visualization
    plt.figure(figsize=(10, 8))

    if labels is not None:
        # Colored by label
        unique_labels = np.unique(labels)
        colors = ['#FF6B6B', '#4ECDC4']  # Red for world, teal for endgrain
        label_names = ['World (Non-wood)', 'End-grain']

        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=colors[i],
                label=label_names[i],
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )

        plt.legend(fontsize=12, framealpha=0.9)
    else:
        # No labels
        plt.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c='steelblue',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.title(f'Latent Space Visualization ({method.upper()})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def visualize_reconstruction(
    model,
    dataset,
    indices,
    device='cuda',
    output_path=None
):
    """
    Visualize feature reconstruction quality

    Args:
        model: Trained autoencoder
        dataset: Feature dataset
        indices: Indices of samples to visualize
        device: Device to use
        output_path: Path to save plot
    """
    model.eval()
    model = model.to(device)

    n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))

    if n_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get features
            if isinstance(dataset[idx], tuple):
                features, label = dataset[idx]
                label_text = "End-grain" if label == 1 else "World"
            else:
                features = dataset[idx]
                label_text = "Unknown"

            features = features.unsqueeze(0).to(device)

            # Reconstruct
            if isinstance(model, VariationalAutoencoder):
                reconstructed, _, _ = model(features)
            else:
                reconstructed, _ = model(features)

            # Plot
            features_np = features.cpu().numpy().flatten()
            reconstructed_np = reconstructed.cpu().numpy().flatten()

            x = np.arange(len(features_np))
            axes[i].plot(x, features_np, label='Original', alpha=0.7, linewidth=1.5)
            axes[i].plot(x, reconstructed_np, label='Reconstructed', alpha=0.7, linewidth=1.5)

            mse = np.mean((features_np - reconstructed_np) ** 2)
            axes[i].set_title(f'Sample {idx} ({label_text}) - MSE: {mse:.6f}', fontweight='bold')
            axes[i].set_xlabel('Feature Index')
            axes[i].set_ylabel('Feature Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize autoencoder latent space')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--endgrain_dir', type=str, required=True,
                       help='Directory with end-grain images')
    parser.add_argument('--world_dir', type=str, required=True,
                       help='Directory with world images')
    parser.add_argument('--output_dir', type=str, default='results/latent_viz',
                       help='Output directory')
    parser.add_argument('--method', type=str, default='all',
                       choices=['pca', 'tsne', 'umap', 'all'],
                       help='Visualization method')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Latent Space Visualization")
    print("=" * 80)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Determine model type from checkpoint
    state_dict = checkpoint['model_state_dict']
    has_fc_mu = any('fc_mu' in key for key in state_dict.keys())
    model_type = 'variational' if has_fc_mu else 'standard'

    print(f"Model type: {model_type}")

    # Get model architecture from state dict
    input_dim = state_dict['encoder.0.weight'].shape[1]
    if has_fc_mu:
        latent_dim = state_dict['fc_mu.weight'].shape[0]
        model = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    else:
        # Find latent dim from last encoder layer
        encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.') and 'weight' in k]
        last_key = sorted(encoder_keys)[-1]
        latent_dim = state_dict[last_key].shape[0]
        model = FeatureAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    print(f"Input dim: {input_dim}, Latent dim: {latent_dim}")

    # Initialize feature extractor
    print("\nInitializing feature extractor...")
    extractor = EndGrainFeatureExtractor(normalize=True)

    # Load images
    print("\nLoading images...")
    endgrain_paths = load_image_paths_from_directory(args.endgrain_dir)
    world_paths = load_image_paths_from_directory(args.world_dir)

    # Sample if too many
    np.random.seed(args.seed)
    if len(endgrain_paths) > args.n_samples // 2:
        endgrain_paths = list(np.random.choice(endgrain_paths, args.n_samples // 2, replace=False))
    if len(world_paths) > args.n_samples // 2:
        world_paths = list(np.random.choice(world_paths, args.n_samples // 2, replace=False))

    # Create dataset
    all_paths = endgrain_paths + world_paths
    labels = [1] * len(endgrain_paths) + [0] * len(world_paths)

    print(f"Using {len(endgrain_paths)} end-grain and {len(world_paths)} world images")

    dataset = FeatureDataset(
        all_paths,
        extractor,
        labels=labels,
        cache_features=True
    )

    # Precompute features
    print("\nExtracting features...")
    dataset.precompute_features()

    # Extract latent representations
    latent_vectors, labels_array = extract_latent_representations(
        model,
        dataset,
        device=args.device,
        batch_size=args.batch_size
    )

    print(f"Latent vectors shape: {latent_vectors.shape}")

    # Visualize
    methods = ['pca', 'tsne', 'umap'] if args.method == 'all' else [args.method]

    for method in methods:
        visualize_latent_space_2d(
            latent_vectors,
            labels_array,
            method=method,
            output_path=output_dir / f'latent_space_{method}.png'
        )

    # Visualize reconstruction for a few samples
    print("\nVisualizing reconstructions...")
    sample_indices = [0, 10, len(endgrain_paths), len(endgrain_paths) + 10]
    visualize_reconstruction(
        model,
        dataset,
        sample_indices,
        device=args.device,
        output_path=output_dir / 'reconstructions.png'
    )

    print(f"\nVisualization complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
