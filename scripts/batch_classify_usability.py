#!/usr/bin/env python3
"""
Batch classify wood images as usable vs unusable

Usage:
    python scripts/batch_classify_usability.py --model models/usability/best_model.pt --input-dir outputs/wood_classified/wood/ --output-dir outputs/usability_classified/
    python scripts/batch_classify_usability.py --model-type rf --model models/usability/blob_ml/ --input-dir outputs/wood_classified/wood/ --output-dir outputs/usability_classified/ --use-texture
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import WoodFeatureExtractor
from src.models import WoodRandomForest, ResNet18, EfficientNetClassifier
from src.data.transforms import get_test_transforms


def load_model(model_type, model_path, device='cpu', use_texture=False, vol_threshold=900, cnn_arch='efficientnet_b0'):
    """Load model based on type"""
    if model_type == 'rf':
        model = WoodRandomForest.load(Path(model_path))
        feature_extractor = WoodFeatureExtractor(
            vol_threshold=vol_threshold,
            use_texture=use_texture
        )
        return model, feature_extractor, None

    elif model_type == 'mlp':
        checkpoint = torch.load(model_path, map_location=device)
        from src.models.mlp import WoodMLP

        input_dim = checkpoint['model_config']['input_dim']
        model = WoodMLP(input_dim=input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        scaler = checkpoint['scaler']
        feature_extractor = WoodFeatureExtractor(
            vol_threshold=vol_threshold,
            use_texture=use_texture
        )
        return (model, scaler), feature_extractor, None

    elif model_type == 'cnn':
        checkpoint = torch.load(model_path, map_location=device)

        # Determine architecture from checkpoint or use specified architecture
        if 'model.features' in str(checkpoint['model_state_dict'].keys()):
            # EfficientNet architecture
            model = EfficientNetClassifier(num_classes=2, model_name=cnn_arch, pretrained=False)
        else:
            # ResNet architecture
            model = ResNet18(num_classes=2)

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model, None, get_test_transforms()

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_with_features(image_path, model, feature_extractor, model_type, skip_vol_check=True):
    """Predict using feature-based model"""
    try:
        features = feature_extractor.extract(image_path)

        # For usability task, we typically skip VoL check since wood images are already filtered
        if not skip_vol_check and features['is_clear'] == 0:
            return {
                'prediction': 'blurry',
                'confidence': 0.0,
                'usable_probability': 0.0,
                'vol_score': features['vol_score'],
                'is_clear': False
            }

        # Prepare features
        feature_cols = [k for k in features.keys() if k not in ['vol_score', 'is_clear']]
        X = np.array([[features[k] for k in feature_cols]])

        # Predict
        if model_type == 'rf':
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
        else:  # MLP
            model_obj, scaler = model
            X_scaled = scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            with torch.no_grad():
                prob_usable = model_obj(X_tensor).item()
            pred = 1 if prob_usable > 0.5 else 0
            prob = [1 - prob_usable, prob_usable]

        label = 'usable' if pred == 1 else 'unusable'
        confidence = prob[pred]

        return {
            'prediction': label,
            'confidence': confidence,
            'usable_probability': prob[1],
            'vol_score': features.get('vol_score', None),
            'is_clear': True
        }
    except Exception as e:
        return {
            'prediction': 'error',
            'confidence': 0.0,
            'usable_probability': 0.0,
            'vol_score': None,
            'is_clear': False,
            'error': str(e)
        }


def predict_with_cnn(image_path, model, transform, device):
    """Predict using CNN model"""
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform and predict
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        pred = np.argmax(probs)
        label = 'usable' if pred == 1 else 'unusable'

        return {
            'prediction': label,
            'confidence': probs[pred],
            'usable_probability': probs[1],
            'vol_score': None,
            'is_clear': True
        }
    except Exception as e:
        return {
            'prediction': 'error',
            'confidence': 0.0,
            'usable_probability': 0.0,
            'vol_score': None,
            'is_clear': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Batch classify wood images as usable vs unusable')

    # Model arguments
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['rf', 'mlp', 'cnn'],
                       help='Model type')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file or directory')

    # Input/output
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing wood images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory to organize classified images')
    parser.add_argument('--csv-output', type=str, default=None,
                       help='CSV file to save results (default: output_dir/results.csv)')

    # Feature extraction (for RF/MLP)
    parser.add_argument('--use-texture', action='store_true',
                       help='Use texture features (for RF/MLP)')
    parser.add_argument('--vol-threshold', type=float, default=900,
                       help='VoL threshold for blur detection (default: 900)')
    parser.add_argument('--skip-vol-check', action='store_true', default=True,
                       help='Skip VoL blur check (default: True for usability task)')

    # CNN arguments
    parser.add_argument('--cnn-arch', type=str, default='efficientnet_b0',
                       choices=['resnet18', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                               'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                               'efficientnet_b6', 'efficientnet_b7'],
                       help='CNN architecture (default: efficientnet_b0)')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size for CNN (default: 224)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto-detect)')

    # Options
    parser.add_argument('--copy-files', action='store_true',
                       help='Copy images to output subdirectories (usable/, unusable/, error/)')
    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                       help='Minimum confidence threshold (default: 0.0, no filtering)')

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv_output:
        csv_path = Path(args.csv_output)
    else:
        csv_path = output_dir / 'results.csv'

    # Setup device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print("=" * 80)
    print(f"BATCH USABILITY CLASSIFICATION")
    print("=" * 80)
    print(f"Model: {args.model_type.upper()}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Skip VoL check: {args.skip_vol_check}")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {args.model}")
    if args.model_type in ['rf', 'mlp']:
        model, feature_extractor, transform = load_model(
            args.model_type, args.model, device, args.use_texture, args.vol_threshold
        )
    else:
        model, feature_extractor, transform = load_model(
            args.model_type, args.model, device, cnn_arch=args.cnn_arch
        )

    # Create output subdirectories if copying files
    if args.copy_files:
        (output_dir / 'usable').mkdir(exist_ok=True)
        (output_dir / 'unusable').mkdir(exist_ok=True)
        (output_dir / 'error').mkdir(exist_ok=True)

    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(input_dir.glob(ext)))

    print(f"\nFound {len(image_files)} images")
    print("\nProcessing...")

    # Process images
    results = []
    for img_path in tqdm(image_files):
        if args.model_type in ['rf', 'mlp']:
            result = predict_with_features(img_path, model, feature_extractor, args.model_type, args.skip_vol_check)
        else:
            result = predict_with_cnn(img_path, model, transform, device)

        result['filename'] = img_path.name
        result['filepath'] = str(img_path)
        results.append(result)

        # Copy files to appropriate subdirectories
        if args.copy_files and result['prediction'] in ['usable', 'unusable', 'error']:
            dest = output_dir / result['prediction'] / img_path.name
            shutil.copy2(img_path, dest)

    # Save results to CSV
    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['filename', 'prediction', 'confidence', 'usable_probability']
    if 'vol_score' in df.columns:
        cols.append('vol_score')
    cols.extend([c for c in df.columns if c not in cols])
    df = df[cols]

    df.to_csv(csv_path, index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total images: {len(results)}")
    print(f"\nPrediction counts:")
    print(df['prediction'].value_counts().to_string())

    if args.confidence_threshold > 0:
        high_conf = df[df['confidence'] >= args.confidence_threshold]
        print(f"\nHigh confidence (>= {args.confidence_threshold}): {len(high_conf)} ({len(high_conf)/len(df)*100:.1f}%)")

    print(f"\nResults saved to: {csv_path}")
    if args.copy_files:
        print(f"Files organized in: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
