#!/usr/bin/env python3
"""
Batch classify wood images as usable vs unusable using CNN model.

Usage:
    python scripts/batch_classify_usability.py --model models/usability/best_model.pt --input-dir images/ --output-dir classified/
    python scripts/batch_classify_usability.py --model models/usability/best_model.pt --input-dir images/ --output-dir classified/ --threshold 0.6 --copy-files
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ResNet18, EfficientNetClassifier
from src.data.transforms import get_test_transforms


def load_cnn_model(model_path, device='cpu'):
    """Load CNN model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Auto-detect num_classes and in_channels from checkpoint
    num_classes = 1
    in_channels = 3
    model_type = 'efficientnet'

    if 'model.classifier.1.weight' in state_dict:
        num_classes = state_dict['model.classifier.1.weight'].shape[0]
        in_channels = state_dict['model.features.0.0.weight'].shape[1]
        model_type = 'efficientnet'
    elif 'fc.weight' in state_dict:
        num_classes = state_dict['fc.weight'].shape[0]
        if 'conv1.weight' in state_dict:
            in_channels = state_dict['conv1.weight'].shape[1]
        model_type = 'resnet'

    print(f"  Model type: {model_type}")
    print(f"  Num classes: {num_classes}")
    print(f"  Input channels: {in_channels} ({'grayscale' if in_channels == 1 else 'RGB'})")

    if model_type == 'efficientnet':
        model = EfficientNetClassifier(num_classes=num_classes, in_channels=in_channels)
    else:
        model = ResNet18(num_classes=num_classes, in_channels=in_channels)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    transform = get_test_transforms(grayscale=(in_channels == 1))
    return model, transform, num_classes


def predict_image(image_path, model, transform, device, threshold=0.5, num_classes=1):
    """Predict usability probability for a single image."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Failed to load image")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)

            if num_classes == 1:
                prob_usable = torch.sigmoid(outputs).squeeze().cpu().item()
                pred = 1 if prob_usable >= threshold else 0
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                prob_usable = probs[1]
                pred = 1 if prob_usable >= threshold else 0

        label = 'usable' if pred == 1 else 'unusable'
        confidence = prob_usable if pred == 1 else (1 - prob_usable)

        return {
            'prediction': label,
            'confidence': confidence,
            'usable_probability': prob_usable
        }
    except Exception as e:
        return {
            'prediction': 'error',
            'confidence': 0.0,
            'usable_probability': 0.0,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Batch classify wood images as usable vs unusable')

    parser.add_argument('--model', type=str, required=True,
                        help='Path to CNN model checkpoint (.pt file)')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--csv-output', type=str, default=None,
                        help='CSV file path (default: output_dir/results.csv)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--copy-files', action='store_true',
                        help='Copy images to usable/ and unusable/ subdirectories')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto-detect)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv_output) if args.csv_output else output_dir / 'results.csv'

    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print("=" * 70)
    print("BATCH USABILITY CLASSIFICATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}")
    print("=" * 70)

    print("\nLoading model...")
    model, transform, num_classes = load_cnn_model(args.model, device)

    if args.copy_files:
        (output_dir / 'usable').mkdir(exist_ok=True)
        (output_dir / 'unusable').mkdir(exist_ok=True)
        (output_dir / 'error').mkdir(exist_ok=True)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(input_dir.glob(ext)))

    print(f"\nFound {len(image_files)} images")
    print("Processing...")

    results = []
    for img_path in tqdm(image_files):
        result = predict_image(img_path, model, transform, device, args.threshold, num_classes)
        result['filename'] = img_path.name
        result['filepath'] = str(img_path)
        results.append(result)

        if args.copy_files:
            dest_dir = output_dir / result['prediction']
            shutil.copy2(img_path, dest_dir / img_path.name)

    df = pd.DataFrame(results)
    cols = ['filename', 'prediction', 'confidence', 'usable_probability']
    cols.extend([c for c in df.columns if c not in cols])
    df = df[cols]
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total images: {len(results)}")
    print(f"\nPrediction counts:")
    print(df['prediction'].value_counts().to_string())
    print(f"\nResults saved to: {csv_path}")
    if args.copy_files:
        print(f"Files organized in: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
