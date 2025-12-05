#!/usr/bin/env python3
"""
Run inference on single image or directory

Usage:
    # Single image
    python scripts/predict.py --model-type cnn --model models/cnn/best_model.pt --image test.jpg

    # Directory (batch)
    python scripts/predict.py --model-type rf --model models/blob_ml/ --input-dir test_images/ --output results.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import WoodFeatureExtractor
from src.models import WoodRandomForest, load_mlp_model, ResNet18, EfficientNetClassifier
from src.data.transforms import get_test_transforms


def predict_blob_ml(model, feature_extractor, image_path):
    """Predict using blob features + ML model"""
    features = feature_extractor.extract(image_path)

    if features['is_clear'] == 0:
        return {
            'prediction': 'BLURRY',
            'confidence': 0.0,
            'vol_score': features['vol_score']
        }

    # Prepare features
    feature_cols = [k for k in features.keys() if k not in ['vol_score', 'is_clear']]
    X = np.array([[features[k] for k in feature_cols]])

    # Predict
    if isinstance(model, WoodRandomForest):
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
    else:  # MLP
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            prob_wood = model(X_tensor).item()
        pred = 1 if prob_wood > 0.5 else 0
        prob = [1 - prob_wood, prob_wood]

    label = 'WOOD' if pred == 1 else 'NON-WOOD'
    confidence = prob[pred]

    return {
        'prediction': label,
        'confidence': confidence,
        'wood_probability': prob[1],
        'vol_score': features['vol_score']
    }


def predict_cnn(model, transform, image_path, device):
    """Predict using CNN model"""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transform
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred = np.argmax(probs)
    label = 'WOOD' if pred == 1 else 'NON-WOOD'
    confidence = probs[pred]

    return {
        'prediction': label,
        'confidence': confidence,
        'wood_probability': probs[1]
    }


def main():
    parser = argparse.ArgumentParser(description='Run inference on wood images')

    # Model arguments
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['rf', 'mlp', 'cnn'],
                       help='Model type: rf, mlp, or cnn')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file or directory')

    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str,
                       help='Single image file')
    group.add_argument('--input-dir', type=str,
                       help='Directory of images')

    # Output arguments
    parser.add_argument('--output', type=str,
                       help='Output CSV file (for batch prediction)')

    # Feature extraction (for blob ML models)
    parser.add_argument('--use-texture', action='store_true',
                       help='Use texture features (for blob ML models)')
    parser.add_argument('--vol-threshold', type=float, default=900,
                       help='VoL threshold for blur detection (default: 900)')

    # CNN arguments
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size for CNN (default: 224)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto-detect)')

    args = parser.parse_args()

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
    print(f"Running inference with {args.model_type.upper()} model")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {args.model}")

    if args.model_type == 'rf':
        model = WoodRandomForest.load(Path(args.model))
        feature_extractor = WoodFeatureExtractor(
            vol_threshold=args.vol_threshold,
            use_texture=args.use_texture
        )
        predict_fn = lambda img: predict_blob_ml(model, feature_extractor, img)

    elif args.model_type == 'mlp':
        checkpoint = torch.load(args.model, map_location=device)
        from sklearn.preprocessing import StandardScaler

        # Recreate model
        from src.models.mlp import WoodMLP
        input_dim = checkpoint['model_config']['input_dim']
        model = WoodMLP(input_dim=input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # Get scaler
        scaler = checkpoint['scaler']

        feature_extractor = WoodFeatureExtractor(
            vol_threshold=args.vol_threshold,
            use_texture=args.use_texture
        )

        def predict_mlp_wrapped(img):
            features = feature_extractor.extract(img)
            if features['is_clear'] == 0:
                return {'prediction': 'BLURRY', 'confidence': 0.0, 'vol_score': features['vol_score']}

            feature_cols = [k for k in features.keys() if k not in ['vol_score', 'is_clear']]
            X = np.array([[features[k] for k in feature_cols]])
            X_scaled = scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(device)

            with torch.no_grad():
                prob_wood = model(X_tensor).item()

            pred = 1 if prob_wood > 0.5 else 0
            label = 'WOOD' if pred == 1 else 'NON-WOOD'

            return {
                'prediction': label,
                'confidence': prob_wood if pred == 1 else 1 - prob_wood,
                'wood_probability': prob_wood,
                'vol_score': features['vol_score']
            }

        predict_fn = predict_mlp_wrapped

    else:  # CNN
        # Load CNN model
        checkpoint = torch.load(args.model, map_location=device)

        # Infer model architecture from checkpoint
        # (You may need to specify this explicitly)
        model = ResNet18(num_classes=2)  # Assume ResNet18 for now
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        transform = get_test_transforms(args.image_size)
        predict_fn = lambda img: predict_cnn(model, transform, img, device)

    # Run inference
    if args.image:
        # Single image prediction
        print(f"\nPredicting on: {args.image}")
        result = predict_fn(args.image)

        print("\n" + "=" * 80)
        print("RESULT")
        print("=" * 80)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        if 'wood_probability' in result:
            print(f"Wood Probability: {result['wood_probability']:.4f}")
        if 'vol_score' in result:
            print(f"VoL Score: {result['vol_score']:.2f}")
        print("=" * 80)

    else:
        # Batch prediction
        input_dir = Path(args.input_dir)
        image_files = list(input_dir.glob('*.jpg')) + \
                     list(input_dir.glob('*.jpeg')) + \
                     list(input_dir.glob('*.png'))

        print(f"\nProcessing {len(image_files)} images from {input_dir}")

        results = []
        for img_path in tqdm(image_files):
            try:
                result = predict_fn(str(img_path))
                results.append({
                    'filename': img_path.name,
                    **result
                })
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                results.append({
                    'filename': img_path.name,
                    'prediction': 'ERROR',
                    'confidence': 0.0
                })

        # Save results
        df = pd.DataFrame(results)

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
        else:
            print("\n" + "=" * 80)
            print("RESULTS")
            print("=" * 80)
            print(df.to_string())

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(df['prediction'].value_counts())
        print("=" * 80)


if __name__ == '__main__':
    main()
