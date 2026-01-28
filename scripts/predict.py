#!/usr/bin/env python3
"""
Run inference on a single image or directory using CNN model.

Usage:
    # Single image
    python scripts/predict.py --model models/cnn/best_model.pt --image test.jpg

    # Batch (directory)
    python scripts/predict.py --model models/cnn/best_model.pt --input-dir images/ --output results.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm

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


def predict_image(image_path, model, transform, device, num_classes=1):
    """Predict class for a single image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        if num_classes == 1:
            prob_positive = torch.sigmoid(outputs).squeeze().cpu().item()
            pred = 1 if prob_positive >= 0.5 else 0
        else:
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prob_positive = probs[1]
            pred = np.argmax(probs)

    label = 'POSITIVE' if pred == 1 else 'NEGATIVE'
    confidence = prob_positive if pred == 1 else (1 - prob_positive)

    return {
        'prediction': label,
        'confidence': confidence,
        'positive_probability': prob_positive
    }


def main():
    parser = argparse.ArgumentParser(description='Run CNN inference on images')

    parser.add_argument('--model', type=str, required=True,
                        help='Path to CNN model checkpoint (.pt file)')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Single image file')
    group.add_argument('--input-dir', type=str, help='Directory of images')

    parser.add_argument('--output', type=str, help='Output CSV file (for batch)')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto-detect)')

    args = parser.parse_args()

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
    print("CNN INFERENCE")
    print("=" * 70)

    print(f"\nLoading model: {args.model}")
    model, transform, num_classes = load_cnn_model(args.model, device)

    if args.image:
        # Single image prediction
        print(f"\nPredicting: {args.image}")
        result = predict_image(args.image, model, transform, device, num_classes)

        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Positive Probability: {result['positive_probability']:.4f}")
        print("=" * 70)

    else:
        # Batch prediction
        input_dir = Path(args.input_dir)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(input_dir.glob(ext)))

        print(f"\nProcessing {len(image_files)} images from {input_dir}")

        results = []
        for img_path in tqdm(image_files):
            try:
                result = predict_image(str(img_path), model, transform, device, num_classes)
                results.append({'filename': img_path.name, **result})
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                results.append({
                    'filename': img_path.name,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'positive_probability': 0.0
                })

        df = pd.DataFrame(results)

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
        else:
            print("\n" + "=" * 70)
            print("RESULTS")
            print("=" * 70)
            print(df.to_string())

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(df['prediction'].value_counts())
        print("=" * 70)


if __name__ == '__main__':
    main()
