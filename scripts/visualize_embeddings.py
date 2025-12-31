"""
Visualize learned embeddings in 2D/3D space

This script provides geometric visualization of how the contrastive learning
model organizes samples in the embedding space.

Visualization methods:
1. t-SNE: Non-linear dimensionality reduction (preserves local structure)
2. UMAP: Similar to t-SNE but faster and preserves global structure
3. PCA: Linear dimensionality reduction (shows principal components)
4. Cosine similarity heatmap
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import WoodImageDataset
from src.data.transforms import get_val_transforms
from src.contrastive import SimCLRModel


def extract_embeddings(model, dataloader, device, use_projection=False):
    """
    Extract embeddings for all samples in the dataloader

    Args:
        model: Trained SimCLR model
        dataloader: DataLoader
        device: Device
        use_projection: If True, use projection head output; else use encoder output

    Returns:
        embeddings: (N, D) numpy array
        labels: (N,) numpy array
        image_paths: List of image paths
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extracting embeddings'):
            images = images.to(device)

            if use_projection:
                # Use projection head output (used during training)
                _, embeddings = model(images)
            else:
                # Use encoder output (used for downstream tasks)
                embeddings = model.get_representation(images)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return embeddings, labels


def visualize_tsne(embeddings, labels, title="t-SNE Visualization", save_path=None):
    """
    Visualize embeddings using t-SNE

    Args:
        embeddings: (N, D) numpy array
        labels: (N,) numpy array
        title: Plot title
        save_path: Path to save figure
    """
    print("\nComputing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot each class with different color
    unique_labels = np.unique(labels)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[idx % len(colors)],
            label=f'Class {label}',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE visualization to {save_path}")

    return embeddings_2d


def visualize_umap(embeddings, labels, title="UMAP Visualization", save_path=None):
    """
    Visualize embeddings using UMAP

    Args:
        embeddings: (N, D) numpy array
        labels: (N,) numpy array
        title: Plot title
        save_path: Path to save figure
    """
    try:
        import umap
    except ImportError:
        print("⚠ UMAP not installed. Install with: pip install umap-learn")
        return None

    print("\nComputing UMAP projection...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Create plot
    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[idx % len(colors)],
            label=f'Class {label}',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved UMAP visualization to {save_path}")

    return embeddings_2d


def visualize_pca(embeddings, labels, title="PCA Visualization", save_path=None):
    """
    Visualize embeddings using PCA

    Args:
        embeddings: (N, D) numpy array
        labels: (N,) numpy array
        title: Plot title
        save_path: Path to save figure
    """
    print("\nComputing PCA projection...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Create plot
    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[idx % len(colors)],
            label=f'Class {label}',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA visualization to {save_path}")

    # Plot explained variance
    if pca.n_components_ > 2:
        plt.figure(figsize=(8, 5))
        variance_ratio = pca.explained_variance_ratio_[:10]
        plt.bar(range(1, len(variance_ratio) + 1), variance_ratio)
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Explained Variance Ratio', fontsize=12)
        plt.title('PCA Explained Variance', fontsize=14)
        plt.tight_layout()

        if save_path:
            variance_path = save_path.replace('.png', '_variance.png')
            plt.savefig(variance_path, dpi=300, bbox_inches='tight')

    return embeddings_2d


def visualize_similarity_matrix(embeddings, labels, max_samples=200, save_path=None):
    """
    Visualize cosine similarity matrix

    Shows how similar samples are to each other in embedding space.
    Ideally, samples from the same class should have high similarity.

    Args:
        embeddings: (N, D) numpy array
        labels: (N,) numpy array
        max_samples: Maximum number of samples to visualize (for readability)
        save_path: Path to save figure
    """
    print("\nComputing cosine similarity matrix...")

    # Subsample if too many samples
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute cosine similarity matrix
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)

    # Sort by labels for better visualization
    sorted_indices = np.argsort(labels)
    similarity_matrix_sorted = similarity_matrix[sorted_indices][:, sorted_indices]
    labels_sorted = labels[sorted_indices]

    # Create plot
    plt.figure(figsize=(12, 10))

    # Plot heatmap
    sns.heatmap(
        similarity_matrix_sorted,
        cmap='RdYlBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'},
        xticklabels=False,
        yticklabels=False
    )

    plt.title('Cosine Similarity Matrix (sorted by class)', fontsize=14, fontweight='bold')
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Samples', fontsize=12)

    # Add class boundaries
    unique_labels = np.unique(labels_sorted)
    boundaries = []
    for label in unique_labels[:-1]:
        boundary = np.where(labels_sorted == label)[0][-1] + 0.5
        boundaries.append(boundary)
        plt.axhline(boundary, color='black', linewidth=2)
        plt.axvline(boundary, color='black', linewidth=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved similarity matrix to {save_path}")


def visualize_3d(embeddings, labels, method='tsne', save_path=None):
    """
    3D visualization of embeddings

    Args:
        embeddings: (N, D) numpy array
        labels: (N,) numpy array
        method: 'tsne', 'umap', or 'pca'
        save_path: Path to save figure
    """
    print(f"\nComputing 3D {method.upper()} projection...")

    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=3, random_state=42, perplexity=30)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=3, random_state=42)
        except ImportError:
            print("⚠ UMAP not installed")
            return
    elif method == 'pca':
        reducer = PCA(n_components=3)
    else:
        raise ValueError(f"Unknown method: {method}")

    embeddings_3d = reducer.fit_transform(embeddings)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embeddings_3d[mask, 0],
            embeddings_3d[mask, 1],
            embeddings_3d[mask, 2],
            c=colors[idx % len(colors)],
            label=f'Class {label}',
            alpha=0.6,
            s=30,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_title(f'3D {method.upper()} Visualization', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.set_zlabel(f'{method.upper()} Dimension 3', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D visualization to {save_path}")

    return embeddings_3d


def compute_separation_metrics(embeddings, labels):
    """
    Compute quantitative metrics for class separation

    Args:
        embeddings: (N, D) numpy array
        labels: (N,) numpy array

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    print("\nComputing separation metrics...")

    metrics = {}

    # Silhouette score: [-1, 1], higher is better (measures cluster cohesion and separation)
    metrics['silhouette'] = silhouette_score(embeddings, labels)

    # Davies-Bouldin index: [0, inf), lower is better (ratio of within-cluster to between-cluster distances)
    metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)

    # Calinski-Harabasz index: [0, inf), higher is better (ratio of between-cluster to within-cluster variance)
    metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)

    # Intra-class and inter-class distances
    unique_labels = np.unique(labels)
    intra_distances = []
    inter_distances = []

    # Normalize embeddings for cosine similarity
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    for label in unique_labels:
        mask = labels == label
        class_embeddings = embeddings_norm[mask]

        # Intra-class: average distance within class
        if len(class_embeddings) > 1:
            intra_sim = np.dot(class_embeddings, class_embeddings.T)
            intra_dist = 1 - intra_sim[np.triu_indices_from(intra_sim, k=1)]
            intra_distances.extend(intra_dist)

        # Inter-class: average distance to other classes
        other_embeddings = embeddings_norm[~mask]
        if len(other_embeddings) > 0:
            inter_sim = np.dot(class_embeddings, other_embeddings.T)
            inter_dist = 1 - inter_sim.flatten()
            inter_distances.extend(inter_dist)

    metrics['avg_intra_distance'] = np.mean(intra_distances)
    metrics['avg_inter_distance'] = np.mean(inter_distances)
    metrics['separation_ratio'] = metrics['avg_inter_distance'] / (metrics['avg_intra_distance'] + 1e-8)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Visualize learned embeddings')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--encoder', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50', 'resnet101', 'efficientnet_b0'],
                        help='Encoder architecture')
    parser.add_argument('--grayscale', action='store_true',
                        help='Use grayscale images')
    parser.add_argument('--use_projection', action='store_true',
                        help='Use projection head output instead of encoder output')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing positive/ and negative/ subdirectories')
    parser.add_argument('--positive_dir', type=str, default='positive',
                        help='Positive class directory name')
    parser.add_argument('--negative_dir', type=str, default='negative',
                        help='Negative class directory name')

    # Visualization arguments
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['tsne', 'umap', 'pca'],
                        choices=['tsne', 'umap', 'pca', 'similarity', '3d'],
                        help='Visualization methods to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Embedding Visualization")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Setup device
    device = torch.device(args.device)

    # Load model
    print("\nLoading model...")
    model = SimCLRModel(
        encoder_name=args.encoder,
        pretrained=False,
        grayscale=args.grayscale
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load dataset
    print("\nLoading dataset...")
    data_root = Path(args.data_dir)
    transform = get_val_transforms(grayscale=args.grayscale)

    dataset = WoodImageDataset.from_directories(
        positive_dir=data_root / args.positive_dir,
        negative_dir=data_root / args.negative_dir,
        transform=transform,
        grayscale=args.grayscale
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Extract embeddings
    embeddings, labels = extract_embeddings(
        model, dataloader, device, use_projection=args.use_projection
    )

    print(f"\nExtracted embeddings: {embeddings.shape}")
    print(f"Number of samples per class: {np.bincount(labels)}")

    # Compute separation metrics
    metrics = compute_separation_metrics(embeddings, labels)
    print("\n" + "=" * 60)
    print("Separation Metrics:")
    print("=" * 60)
    print(f"Silhouette Score:         {metrics['silhouette']:.4f}  (higher is better)")
    print(f"Davies-Bouldin Index:     {metrics['davies_bouldin']:.4f}  (lower is better)")
    print(f"Calinski-Harabasz Index:  {metrics['calinski_harabasz']:.2f}  (higher is better)")
    print(f"Avg Intra-class Distance: {metrics['avg_intra_distance']:.4f}")
    print(f"Avg Inter-class Distance: {metrics['avg_inter_distance']:.4f}")
    print(f"Separation Ratio:         {metrics['separation_ratio']:.4f}  (higher is better)")
    print("=" * 60)

    # Save metrics
    metrics_file = output_dir / 'separation_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("Embedding Separation Metrics\n")
        f.write("=" * 60 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    # Visualizations
    embedding_type = "projection" if args.use_projection else "encoder"

    if 'tsne' in args.methods:
        visualize_tsne(
            embeddings, labels,
            title=f't-SNE Visualization ({embedding_type} embeddings)',
            save_path=output_dir / 'tsne_2d.png'
        )

    if 'umap' in args.methods:
        visualize_umap(
            embeddings, labels,
            title=f'UMAP Visualization ({embedding_type} embeddings)',
            save_path=output_dir / 'umap_2d.png'
        )

    if 'pca' in args.methods:
        visualize_pca(
            embeddings, labels,
            title=f'PCA Visualization ({embedding_type} embeddings)',
            save_path=output_dir / 'pca_2d.png'
        )

    if 'similarity' in args.methods:
        visualize_similarity_matrix(
            embeddings, labels,
            save_path=output_dir / 'similarity_matrix.png'
        )

    if '3d' in args.methods:
        visualize_3d(
            embeddings, labels,
            method='tsne',
            save_path=output_dir / 'tsne_3d.png'
        )

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    # Show plots
    plt.show()


if __name__ == '__main__':
    main()
