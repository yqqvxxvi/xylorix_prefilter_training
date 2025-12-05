#!/usr/bin/env python3
"""
Batch filter images by Variance of Laplacian (VoL) for blur detection

Usage:
    python scripts/batch_filter_vol.py --input-dir dataset/images/ --output-dir outputs/vol_filtered/ --threshold 900
    python scripts/batch_filter_vol.py --input-dir dataset/images/ --output-dir outputs/vol_filtered/ --threshold 900 --copy-files
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.quality import compute_variance_of_laplacian


def main():
    parser = argparse.ArgumentParser(description='Batch filter images by VoL (blur detection)')

    # Input/output
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--csv-output', type=str, default=None,
                       help='CSV file to save results (default: output_dir/vol_results.csv)')

    # VoL parameters
    parser.add_argument('--threshold', type=float, default=900,
                       help='VoL threshold (default: 900). Images below this are considered blurry')

    # Options
    parser.add_argument('--copy-files', action='store_true',
                       help='Copy images to output subdirectories (clear/, blurry/)')

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv_output:
        csv_path = Path(args.csv_output)
    else:
        csv_path = output_dir / 'vol_results.csv'

    print("=" * 80)
    print(f"BATCH VoL FILTERING")
    print("=" * 80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Threshold: {args.threshold}")
    print("=" * 80)

    # Create output subdirectories if copying files
    if args.copy_files:
        (output_dir / 'clear').mkdir(exist_ok=True)
        (output_dir / 'blurry').mkdir(exist_ok=True)

    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(input_dir.glob(ext)))

    print(f"\nFound {len(image_files)} images")
    print("\nProcessing...")

    # Process images
    results = []
    for img_path in tqdm(image_files):
        try:
            # Compute VoL
            vol_score = compute_variance_of_laplacian(str(img_path))
            is_clear = vol_score >= args.threshold

            result = {
                'filename': img_path.name,
                'filepath': str(img_path),
                'vol_score': vol_score,
                'is_clear': is_clear,
                'status': 'clear' if is_clear else 'blurry'
            }

            # Copy files to appropriate subdirectories
            if args.copy_files:
                if is_clear:
                    dest = output_dir / 'clear' / img_path.name
                else:
                    dest = output_dir / 'blurry' / img_path.name
                shutil.copy2(img_path, dest)

        except Exception as e:
            result = {
                'filename': img_path.name,
                'filepath': str(img_path),
                'vol_score': 0.0,
                'is_clear': False,
                'status': 'error',
                'error': str(e)
            }

        results.append(result)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total images: {len(results)}")
    print(f"\nStatus counts:")
    print(df['status'].value_counts().to_string())

    clear_count = df['is_clear'].sum()
    print(f"\nClear images: {clear_count} ({clear_count/len(df)*100:.1f}%)")
    print(f"Blurry images: {len(df) - clear_count} ({(len(df) - clear_count)/len(df)*100:.1f}%)")

    if len(df) > 0:
        vol_scores = df[df['status'] != 'error']['vol_score']
        print(f"\nVoL score statistics:")
        print(f"  Mean: {vol_scores.mean():.2f}")
        print(f"  Median: {vol_scores.median():.2f}")
        print(f"  Min: {vol_scores.min():.2f}")
        print(f"  Max: {vol_scores.max():.2f}")

    print(f"\nResults saved to: {csv_path}")
    if args.copy_files:
        print(f"Files organized in: {output_dir}")
        print(f"  Clear: {output_dir / 'clear'}")
        print(f"  Blurry: {output_dir / 'blurry'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
