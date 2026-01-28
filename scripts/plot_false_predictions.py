#!/usr/bin/env python3
"""
Plot false negative or false positive images from model evaluation results

Usage:
    # Plot false negatives (true pos, predicted neg)
    python scripts/plot_false_predictions.py --json results/thresholding/0.3/endgrain_23012026.json --type fn

    # Plot false positives (true neg, predicted pos)
    python scripts/plot_false_predictions.py --json results/thresholding/0.3/endgrain_23012026.json --type fp

    # Plot all incorrect predictions
    python scripts/plot_false_predictions.py --json results/thresholding/0.3/endgrain_23012026.json --type all

    # Customize grid size and output
    python scripts/plot_false_predictions.py --json results/thresholding/0.3/endgrain_23012026.json --type fn --cols 6 --output false_negatives.png

    # Show plot interactively
    python scripts/plot_false_predictions.py --json results/thresholding/0.3/endgrain_23012026.json --type fn --show
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_predictions(results, prediction_type='fn', cols=5, save_path=None, show=False):
    """
    Plot false predictions in a grid

    Args:
        results: List of prediction results from JSON
        prediction_type: 'fn' (false negatives), 'fp' (false positives), or 'all' (all incorrect)
        cols: Number of columns in the grid
        save_path: Path to save the plot
        show: Whether to show the plot interactively
    """
    # Filter results based on prediction type
    if prediction_type == 'fn':
        # False negatives: true pos, predicted neg
        filtered = [r for r in results if r['true_label'] == 'pos' and r['predicted_label'] == 'neg']
        title_prefix = "False Negatives"
    elif prediction_type == 'fp':
        # False positives: true neg, predicted pos
        filtered = [r for r in results if r['true_label'] == 'neg' and r['predicted_label'] == 'pos']
        title_prefix = "False Positives"
    elif prediction_type == 'all':
        # All incorrect predictions
        filtered = [r for r in results if not r['correct']]
        title_prefix = "All Incorrect Predictions"
    else:
        raise ValueError(f"Invalid prediction_type: {prediction_type}. Must be 'fn', 'fp', or 'all'")

    if len(filtered) == 0:
        print(f"No {title_prefix.lower()} found!")
        return

    print(f"Found {len(filtered)} {title_prefix.lower()}")

    # Calculate grid dimensions
    rows = math.ceil(len(filtered) / cols)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(f"{title_prefix} (n={len(filtered)})", fontsize=16, fontweight='bold')

    # Flatten axes for easier iteration
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Plot each image
    for idx, result in enumerate(filtered):
        ax = axes[idx]

        # Load and display image
        img_path = Path(result['image'])
        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', fontsize=12)

        # Set title with prediction info
        true_label = result['true_label']
        pred_label = result['predicted_label']
        confidence = result['confidence']

        title = f"True: {true_label}\nPred: {pred_label} ({confidence:.3f})"
        ax.set_title(title, fontsize=9)

        # Add filename as xlabel
        ax.set_xlabel(img_path.name, fontsize=7)

        ax.axis('off')

    # Hide unused subplots
    for idx in range(len(filtered), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()

    if not show:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot false predictions from model evaluation JSON')

    # Input arguments
    parser.add_argument('--json', type=str, required=True,
                       help='Path to evaluation results JSON file')
    parser.add_argument('--type', type=str, default='fn', choices=['fn', 'fp', 'all'],
                       help='Type of predictions to plot: fn (false negatives), fp (false positives), or all (default: fn)')

    # Display arguments
    parser.add_argument('--cols', type=int, default=5,
                       help='Number of columns in the grid (default: 5)')

    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: same directory as JSON)')
    parser.add_argument('--show', action='store_true',
                       help='Show plot interactively')

    args = parser.parse_args()

    # Load JSON file
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)

    print(f"Loading evaluation results from: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract detailed results
    if 'detailed_results' not in data:
        print("Error: JSON file does not contain 'detailed_results' field")
        sys.exit(1)

    results = data['detailed_results']

    # Print summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Model: {data.get('model', 'N/A')}")
    print(f"Total samples: {data['metrics']['total_samples']}")
    print(f"Accuracy: {data['metrics']['accuracy']:.4f}")
    print(f"Correct predictions: {data['metrics']['correct_predictions']}")
    print(f"Incorrect predictions: {data['metrics']['total_samples'] - data['metrics']['correct_predictions']}")

    # Count false negatives and false positives
    fn_count = sum(1 for r in results if r['true_label'] == 'pos' and r['predicted_label'] == 'neg')
    fp_count = sum(1 for r in results if r['true_label'] == 'neg' and r['predicted_label'] == 'pos')

    print(f"\nFalse Negatives: {fn_count}")
    print(f"False Positives: {fp_count}")
    print("=" * 80 + "\n")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        type_suffix = {'fn': 'false_negatives', 'fp': 'false_positives', 'all': 'incorrect_predictions'}
        output_path = json_path.parent / f"{json_path.stem}_{type_suffix[args.type]}.png"

    # Generate plot
    plot_predictions(
        results,
        prediction_type=args.type,
        cols=args.cols,
        save_path=output_path if not args.show else None,
        show=args.show
    )

    if not args.show and output_path:
        print(f"\nDone!")


if __name__ == '__main__':
    main()
