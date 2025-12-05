#!/usr/bin/env python3
"""
Plot training history from CSV files

Usage:
    # Plot from a specific CSV file
    python scripts/plot_training.py --csv models/cnn/training_history.csv

    # Plot from a model directory (auto-finds training_history.csv)
    python scripts/plot_training.py --model-dir models/cnn/wood_resnet18_batch32_lr0.001_20251205_143022/

    # Save to custom location
    python scripts/plot_training.py --csv models/cnn/training_history.csv --output my_plot.png

    # Show plot interactively
    python scripts/plot_training.py --csv models/cnn/training_history.csv --show
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.plots import plot_training_history


def main():
    parser = argparse.ArgumentParser(description='Plot training history from CSV')

    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv', type=str,
                      help='Path to training_history.csv file')
    group.add_argument('--model-dir', type=str,
                      help='Path to model directory (auto-finds training_history.csv)')

    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: same directory as CSV with .png extension)')
    parser.add_argument('--show', action='store_true',
                       help='Show plot interactively')

    args = parser.parse_args()

    # Determine CSV path
    if args.csv:
        csv_path = Path(args.csv)
    else:
        model_dir = Path(args.model_dir)
        csv_path = model_dir / 'training_history.csv'

    # Check if CSV exists
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    print(f"Loading training history from: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Validate columns
    required_cols = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in CSV: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Convert to history dictionary
    history = {
        'train_loss': df['train_loss'].tolist(),
        'val_loss': df['val_loss'].tolist(),
        'train_acc': df['train_acc'].tolist(),
        'val_acc': df['val_acc'].tolist()
    }

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = csv_path.parent / 'training_history.png'

    # Print statistics
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total epochs: {len(history['train_loss'])}")
    print(f"\nBest validation loss: {min(history['val_loss']):.4f} (epoch {history['val_loss'].index(min(history['val_loss'])) + 1})")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f} (epoch {history['val_acc'].index(max(history['val_acc'])) + 1})")
    print(f"\nFinal train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_acc'][-1]:.4f}")
    print("=" * 80)

    # Generate plot
    plot_training_history(history, save_path=output_path, show=args.show)

    if not args.show:
        print(f"\nPlot saved to: {output_path}")


if __name__ == '__main__':
    main()
