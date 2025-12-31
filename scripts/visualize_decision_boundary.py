"""
Visualize decision boundaries in 2D embedding space

This script creates beautiful visualizations showing:
1. Decision boundaries (using a classifier trained on embeddings)
2. Sample points color-coded by class
3. Confidence regions (how confident the model is in each region)
4. Misclassified points highlighted
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import WoodImageDataset
from src.data.transforms import get_val_transforms
from src.contrastive import SimCLRModel


def extract_embeddings(model, dataloader, device):
    """Extract embeddings from model"""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extracting embeddings'):
            images = images.to(device)
            embeddings = model.get_representation(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def reduce_dimensions(embeddings, method='tsne'):
    """Reduce embeddings to 2D"""
    print(f"Reducing to 2D using {method.upper()}...")

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        except ImportError:
            print("UMAP not installed, using t-SNE")
            reducer = TSNE(n_components=2, random_state=42)

    return reducer.fit_transform(embeddings)


def plot_decision_boundary(X, y, classifier, title="Decision Boundary", save_path=None):
    """
    Plot decision boundary with confidence regions

    Args:
        X: 2D embeddings (N, 2)
        y: Labels (N,)
        classifier: Trained classifier
        title: Plot title
        save_path: Path to save figure
    """
    # Create a mesh grid
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on the mesh grid
    print("Computing decision boundary...")
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Get prediction probabilities for confidence
    if hasattr(classifier, 'predict_proba'):
        Z_proba = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        # Get max probability for each point (confidence)
        Z_confidence = np.max(Z_proba, axis=1).reshape(xx.shape)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Decision boundary with points
    # Background color by class
    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
    ax1.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)

    # Decision boundary line
    ax1.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0.5])

    # Plot samples
    colors = ['#FF0000', '#0000FF']
    for class_idx in np.unique(y):
        mask = y == class_idx
        ax1.scatter(X[mask, 0], X[mask, 1],
                   c=colors[class_idx], label=f'Class {class_idx}',
                   alpha=0.7, s=60, edgecolors='black', linewidth=1)

    # Highlight misclassified points
    y_pred = classifier.predict(X)
    misclassified = y != y_pred
    if np.any(misclassified):
        ax1.scatter(X[misclassified, 0], X[misclassified, 1],
                   c='yellow', marker='x', s=200, linewidths=3,
                   label=f'Misclassified ({misclassified.sum()})', zorder=10)

    accuracy = accuracy_score(y, y_pred)
    ax1.set_title(f'{title}\nAccuracy: {accuracy*100:.2f}%', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dimension 1', fontsize=12)
    ax1.set_ylabel('Dimension 2', fontsize=12)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Confidence heatmap
    if hasattr(classifier, 'predict_proba'):
        im = ax2.contourf(xx, yy, Z_confidence, levels=20, cmap='RdYlGn', alpha=0.8)
        cbar = plt.colorbar(im, ax=ax2, label='Prediction Confidence')

        # Overlay decision boundary
        ax2.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0.5])

        # Plot samples
        for class_idx in np.unique(y):
            mask = y == class_idx
            ax2.scatter(X[mask, 0], X[mask, 1],
                       c=colors[class_idx], alpha=0.5, s=40,
                       edgecolors='white', linewidth=0.5)

        # Highlight low-confidence regions
        low_confidence = Z_confidence < 0.6
        ax2.contour(xx, yy, low_confidence, colors='orange',
                   linewidths=2, linestyles='dashed', levels=[0.5])

        ax2.set_title('Prediction Confidence\n(Dashed = Low Confidence)',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Dimension 1', fontsize=12)
        ax2.set_ylabel('Dimension 2', fontsize=12)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved decision boundary visualization to {save_path}")

    return fig


def plot_density_and_boundary(X, y, classifier, title="Density + Boundary", save_path=None):
    """
    Plot sample density with decision boundary

    Shows where samples are concentrated in the embedding space.
    """
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Get mesh grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Decision boundary
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plot for each class
    for idx, class_idx in enumerate(np.unique(y)):
        ax = axes[idx]

        # Get samples for this class
        X_class = X[y == class_idx]

        # Compute density using KDE
        if len(X_class) > 1:
            print(f"Computing density for class {class_idx}...")
            kde = gaussian_kde(X_class.T)
            density = kde(np.c_[xx.ravel(), yy.ravel()].T).reshape(xx.shape)

            # Plot density heatmap
            im = ax.contourf(xx, yy, density, levels=20, cmap='YlOrRd', alpha=0.7)
            plt.colorbar(im, ax=ax, label='Sample Density')

        # Plot decision boundary
        ax.contour(xx, yy, Z, colors='blue', linewidths=2, levels=[0.5])

        # Plot samples
        colors = ['#FF0000', '#0000FF']
        ax.scatter(X_class[:, 0], X_class[:, 1],
                  c=colors[class_idx], alpha=0.5, s=30,
                  edgecolors='black', linewidth=0.5)

        ax.set_title(f'Class {class_idx} Density\n(Blue line = Decision Boundary)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Dimension 1', fontsize=10)
        ax.set_ylabel('Dimension 2', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved density visualization to {save_path}")

    return fig


def plot_margin_analysis(X, y, classifier, save_path=None):
    """
    Plot margin analysis - distance from decision boundary

    Points far from the boundary are more confidently classified.
    Points near the boundary are harder examples.
    """
    # Get decision function (distance from boundary)
    if hasattr(classifier, 'decision_function'):
        distances = classifier.decision_function(X)
    else:
        # For classifiers without decision_function, use probability margin
        proba = classifier.predict_proba(X)
        distances = np.abs(proba[:, 1] - 0.5) * 2  # Scale to [0, 1]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Color by margin
    ax1 = axes[0]
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=np.abs(distances),
                         cmap='RdYlGn', s=60, alpha=0.7,
                         edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, ax=ax1, label='Distance from Boundary')

    # Highlight points very close to boundary (hard examples)
    hard_examples = np.abs(distances) < np.percentile(np.abs(distances), 20)
    ax1.scatter(X[hard_examples, 0], X[hard_examples, 1],
               facecolors='none', edgecolors='red', s=150,
               linewidths=2, label='Hard Examples')

    ax1.set_title('Margin Analysis\n(Red circles = Hard examples near boundary)',
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('Dimension 1', fontsize=10)
    ax1.set_ylabel('Dimension 2', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of margins by class
    ax2 = axes[1]
    colors = ['#FF6B6B', '#4ECDC4']
    for class_idx in np.unique(y):
        mask = y == class_idx
        ax2.hist(np.abs(distances[mask]), bins=30, alpha=0.6,
                label=f'Class {class_idx}', color=colors[class_idx],
                edgecolor='black')

    ax2.axvline(np.percentile(np.abs(distances), 20),
               color='red', linestyle='--', linewidth=2,
               label='20th percentile (hard examples)')

    ax2.set_title('Distribution of Margins', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Distance from Decision Boundary', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved margin analysis to {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize decision boundaries')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='resnet50')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--positive_dir', type=str, default='positive')
    parser.add_argument('--negative_dir', type=str, default='negative')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--reduction_method', type=str, default='tsne',
                       choices=['tsne', 'pca', 'umap'])
    parser.add_argument('--output_dir', type=str, default='outputs/decision_boundary')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Decision Boundary Visualization")
    print("=" * 60)

    # Load model
    device = torch.device(args.device)
    model = SimCLRModel(encoder_name=args.encoder, grayscale=args.grayscale).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load data
    transform = get_val_transforms(grayscale=args.grayscale)
    dataset = WoodImageDataset.from_directories(
        positive_dir=Path(args.data_dir) / args.positive_dir,
        negative_dir=Path(args.data_dir) / args.negative_dir,
        transform=transform,
        grayscale=args.grayscale
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Extract embeddings
    embeddings, labels = extract_embeddings(model, dataloader, device)
    print(f"Extracted {len(embeddings)} embeddings")

    # Reduce to 2D
    embeddings_2d = reduce_dimensions(embeddings, method=args.reduction_method)

    # Train simple classifier on 2D embeddings
    print("\nTraining classifier on 2D embeddings...")
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(embeddings_2d, labels)

    # Visualizations
    print("\nCreating visualizations...")

    plot_decision_boundary(
        embeddings_2d, labels, classifier,
        title=f"Decision Boundary ({args.reduction_method.upper()})",
        save_path=output_dir / 'decision_boundary.png'
    )

    plot_density_and_boundary(
        embeddings_2d, labels, classifier,
        title="Sample Density with Decision Boundary",
        save_path=output_dir / 'density_boundary.png'
    )

    plot_margin_analysis(
        embeddings_2d, labels, classifier,
        save_path=output_dir / 'margin_analysis.png'
    )

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    plt.show()


if __name__ == '__main__':
    main()
