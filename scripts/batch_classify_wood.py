#!/usr/bin/env python3
"""
Batch classify images as wood vs non-wood

Usage:
    python scripts/batch_classify_wood.py --model models/cnn/best_model.pt --input-dir dataset/images/ --output-dir outputs/wood_classified/
    python scripts/batch_classify_wood.py --model-type rf --model models/blob_ml/ --input-dir dataset/images/ --output-dir outputs/wood_classified/ --use-texture
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
from src.models import WoodRandomForest, ResNet18
from src.data.transforms import get_test_transforms


def load_model(model_type, model_path, device='cpu', use_texture=False, vol_threshold=900):
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
        state_dict = checkpoint['model_state_dict']

        # Auto-detect num_classes from checkpoint
        num_classes = None
        if 'fc.weight' in state_dict:
            num_classes = state_dict['fc.weight'].shape[0]

        if num_classes is None:
            print("  Warning: Could not detect num_classes, using default=1")
            num_classes = 1

        print(f"  Detected num_classes: {num_classes}")

        model = ResNet18(num_classes=num_classes)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model, None, get_test_transforms()

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_with_features(image_path, model, feature_extractor, model_type):
    """Predict using feature-based model"""
    try:
        features = feature_extractor.extract(image_path)

        if features['is_clear'] == 0:
            return {
                'prediction': 'blurry',
                'confidence': 0.0,
                'wood_probability': 0.0,
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
                prob_wood = model_obj(X_tensor).item()
            pred = 1 if prob_wood > 0.5 else 0
            prob = [1 - prob_wood, prob_wood]

        label = 'wood' if pred == 1 else 'non_wood'
        confidence = prob[pred]

        return {
            'prediction': label,
            'confidence': confidence,
            'wood_probability': prob[1],
            'vol_score': features['vol_score'],
            'is_clear': True
        }
    except Exception as e:
        return {
            'prediction': 'error',
            'confidence': 0.0,
            'wood_probability': 0.0,
            'vol_score': 0.0,
            'is_clear': False,
            'error': str(e)
        }


def predict_with_cnn(image_path, model, transform, device, threshold=0.5):
    """Predict using CNN model

    Args:
        image_path: Path to image
        model: CNN model
        transform: Image transform
        device: Device to use
        threshold: Classification threshold (default: 0.5)
    """
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

            # Handle different output formats
            if outputs.shape[-1] == 1:
                # Single output: binary classification with sigmoid
                prob_wood = torch.sigmoid(outputs).squeeze().cpu().item()
                pred = 1 if prob_wood >= threshold else 0
            else:
                # Two outputs: binary classification with softmax
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                prob_wood = probs[1]
                pred = np.argmax(probs)

        label = 'wood' if pred == 1 else 'non_wood'
        confidence = prob_wood if pred == 1 else (1 - prob_wood)

        return {
            'prediction': label,
            'confidence': confidence,
            'wood_probability': prob_wood,
            'vol_score': None,
            'is_clear': True
        }
    except Exception as e:
        return {
            'prediction': 'error',
            'confidence': 0.0,
            'wood_probability': 0.0,
            'vol_score': None,
            'is_clear': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Batch classify images as wood vs non-wood')

    # Model arguments
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['rf', 'mlp', 'cnn'],
                       help='Model type')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file or directory')

    # Input/output
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory to organize classified images')
    parser.add_argument('--csv-output', type=str, default=None,
                       help='CSV file to save results (default: output_dir/results.csv)')

    # Feature extraction (for RF/MLP)
    parser.add_argument('--use-texture', action='store_true',
                       help='Use texture features (for RF/MLP)')
    parser.add_argument('--vol-threshold', type=float, default=900,
                       help='VoL threshold for blur detection (default: 900)')

    # CNN arguments
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size for CNN (default: 224)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto-detect)')

    # Options
    parser.add_argument('--copy-files', action='store_true',
                       help='Copy images to output subdirectories (wood/, non_wood/, blurry/, error/)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold for binary prediction (default: 0.5)')
    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                       help='Minimum confidence threshold for reporting (default: 0.0, no filtering)')

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
    print(f"BATCH WOOD CLASSIFICATION")
    print("=" * 80)
    print(f"Model: {args.model_type.upper()}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Classification threshold: {args.threshold}")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {args.model}")
    if args.model_type in ['rf', 'mlp']:
        model, feature_extractor, transform = load_model(
            args.model_type, args.model, device, args.use_texture, args.vol_threshold
        )
    else:
        model, feature_extractor, transform = load_model(
            args.model_type, args.model, device
        )

    # Create output subdirectories if copying files
    if args.copy_files:
        (output_dir / 'wood').mkdir(exist_ok=True)
        (output_dir / 'non_wood').mkdir(exist_ok=True)
        (output_dir / 'blurry').mkdir(exist_ok=True)
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
            result = predict_with_features(img_path, model, feature_extractor, args.model_type)
        else:
            result = predict_with_cnn(img_path, model, transform, device, threshold=args.threshold)

        result['filename'] = img_path.name
        result['filepath'] = str(img_path)
        results.append(result)

        # Copy files to appropriate subdirectories
        if args.copy_files:
            prediction = result['prediction']
            if prediction in ['wood', 'non_wood', 'blurry', 'error']:
                dest = output_dir / prediction / img_path.name
                shutil.copy2(img_path, dest)

    # Save results to CSV
    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['filename', 'prediction', 'confidence', 'wood_probability']
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

    if args.model_type in ['rf', 'mlp']:
        clear_images = df[df['is_clear'] == True]
        print(f"\nClear images: {len(clear_images)} ({len(clear_images)/len(df)*100:.1f}%)")

    if args.confidence_threshold > 0:
        high_conf = df[df['confidence'] >= args.confidence_threshold]
        print(f"\nHigh confidence (>= {args.confidence_threshold}): {len(high_conf)} ({len(high_conf)/len(df)*100:.1f}%)")

    print(f"\nResults saved to: {csv_path}")
    if args.copy_files:
        print(f"Files organized in: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
