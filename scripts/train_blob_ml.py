#!/usr/bin/env python3
"""
Train Random Forest or MLP on blob/texture features

Usage:
    python scripts/train_blob_ml.py --model rf --wood-dir dataset/wood/ --non-wood-dir dataset/non_wood/
    python scripts/train_blob_ml.py --model mlp --wood-dir dataset/wood/ --non-wood-dir dataset/non_wood/ --use-texture
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import WoodFeatureExtractor, BLOB_FEATURE_NAMES
from src.models import WoodRandomForest, create_mlp_model
from src.utils.metrics import compute_metrics, print_metrics, print_confusion_matrix


def extract_features_from_images(image_paths, feature_extractor):
    """Extract features from list of image paths"""
    features_list = []
    valid_indices = []

    print(f"Extracting features from {len(image_paths)} images...")
    for idx, img_path in enumerate(tqdm(image_paths)):
        try:
            features = feature_extractor.extract(img_path)
            # Only include if image is clear
            if features['is_clear'] > 0:
                features_list.append(features)
                valid_indices.append(idx)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return features_list, valid_indices


def main():
    parser = argparse.ArgumentParser(description='Train Random Forest or MLP on blob features')

    # Task type
    parser.add_argument('--task', type=str, default='wood',
                       choices=['wood', 'usability'],
                       help='Task: "wood" (wood vs non-wood) or "usability" (usable vs unusable)')

    # Data arguments
    parser.add_argument('--positive-dir', type=str, required=True,
                       help='Positive class directory (wood or usable)')
    parser.add_argument('--negative-dir', type=str, required=True,
                       help='Negative class directory (non-wood or unusable)')

    # Model arguments
    parser.add_argument('--model', type=str, choices=['rf', 'mlp'], default='rf',
                       help='Model type: rf (Random Forest) or mlp (MLP)')

    # Feature arguments
    parser.add_argument('--use-texture', action='store_true',
                       help='Extract texture features (LBP, Gabor, GLCM, FFT)')
    parser.add_argument('--vol-threshold', type=float, default=900,
                       help='VoL threshold for blur detection (default: 900)')

    # Training arguments
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set fraction (default: 0.2)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed (default: 42)')

    # Random Forest specific
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees for Random Forest (default: 100)')
    parser.add_argument('--max-depth', type=int, default=None,
                       help='Max depth of trees (default: None)')

    # MLP specific
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs for MLP (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for MLP (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate for MLP (default: 0.001)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models/blob_ml',
                       help='Output directory for trained model (default: models/blob_ml)')

    args = parser.parse_args()

    # Auto-set class names and VoL check based on task
    if args.task == 'wood':
        args.class_names = ['non_wood', 'wood']
        args.skip_vol_check = False
    else:  # usability
        args.class_names = ['unusable', 'usable']
        args.skip_vol_check = True

    # Auto-generate timestamped directory name
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    texture_tag = "texture" if args.use_texture else "blob"
    auto_name = f"{args.task}_{args.model}_{texture_tag}_{timestamp}"
    args.output_dir = f"{args.output_dir}/{auto_name}"

    # Setup paths
    positive_dir = Path(args.positive_dir)
    negative_dir = Path(args.negative_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Task: {args.task.upper()}")
    print(f"Model: {args.model.upper()}")
    print(f"Features: {'blob+texture' if args.use_texture else 'blob'}")
    print(f"Classes: {args.class_names[0]} (0) vs {args.class_names[1]} (1)")
    print(f"VoL check: {'DISABLED' if args.skip_vol_check else 'ENABLED'}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    # Collect image paths
    positive_images = list(positive_dir.glob('*.jpg')) + list(positive_dir.glob('*.jpeg')) + list(positive_dir.glob('*.png'))
    negative_images = list(negative_dir.glob('*.jpg')) + list(negative_dir.glob('*.jpeg')) + list(negative_dir.glob('*.png'))

    print(f"\nDataset:")
    print(f"  {args.class_names[1]} (positive): {len(positive_images)}")
    print(f"  {args.class_names[0]} (negative): {len(negative_images)}")

    all_images = positive_images + negative_images
    all_labels = [1] * len(positive_images) + [0] * len(negative_images)

    # Create feature extractor
    feature_extractor = WoodFeatureExtractor(
        vol_threshold=args.vol_threshold,
        use_texture=args.use_texture
    )

    # Extract features
    features_list, valid_indices = extract_features_from_images(all_images, feature_extractor)
    valid_labels = [all_labels[i] for i in valid_indices]

    print(f"\nValid images after filtering: {len(features_list)}")

    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    feature_cols = [col for col in df.columns if col not in ['vol_score', 'is_clear']]

    X = df[feature_cols].values
    y = np.array(valid_labels)

    print(f"Feature matrix shape: {X.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )

    print(f"\nSplit:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Train model
    if args.model == 'rf':
        print(f"\nTraining Random Forest (n_estimators={args.n_estimators}, max_depth={args.max_depth})...")

        model = WoodRandomForest(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_seed
        )
        model.fit(X_train, y_train, feature_names=feature_cols)

        # Evaluate
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]

        # Save model
        model.save(output_dir)
        print(f"\nSaved model to {output_dir}/")

        # Feature importance
        importance = model.get_feature_importance()
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 Important Features:")
        for feat, imp in top_features:
            print(f"  {feat}: {imp:.4f}")

    else:  # MLP
        print(f"\nTraining MLP (epochs={args.epochs}, lr={args.lr})...")

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model = create_mlp_model(input_dim=X.shape[1], device=device)

        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Training loop
        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0

            # Mini-batch training
            n_batches = len(X_train_scaled) // args.batch_size
            for i in range(n_batches):
                start_idx = i * args.batch_size
                end_idx = start_idx + args.batch_size

                batch_X = torch.FloatTensor(X_train_scaled[start_idx:end_idx]).to(device)
                batch_y = torch.FloatTensor(y_train[start_idx:end_idx]).unsqueeze(1).to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if epoch % 10 == 0:
                avg_loss = running_loss / n_batches
                print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_probs = model(torch.FloatTensor(X_train_scaled).to(device)).cpu().numpy().flatten()
            test_probs = model(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy().flatten()

        train_preds = (train_probs > 0.5).astype(int)
        test_preds = (test_probs > 0.5).astype(int)

        # Save model and scaler
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {'input_dim': X.shape[1]},
            'scaler': scaler,
            'feature_names': feature_cols
        }, output_dir / 'mlp_model.pt')
        print(f"\nSaved model to {output_dir}/mlp_model.pt")

    # Print results
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)

    train_metrics = compute_metrics(y_train, train_preds, train_probs)
    test_metrics = compute_metrics(y_test, test_preds, test_probs)

    print_metrics(train_metrics, "Train")
    print()
    print_metrics(test_metrics, "Test")

    print_confusion_matrix(y_test, test_preds)

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'split': ['train', 'test'],
        'accuracy': [train_metrics['accuracy'], test_metrics['accuracy']],
        'precision': [train_metrics['precision'], test_metrics['precision']],
        'recall': [train_metrics['recall'], test_metrics['recall']],
        'f1': [train_metrics['f1'], test_metrics['f1']],
        'auc': [train_metrics.get('auc', 0), test_metrics.get('auc', 0)]
    })
    metrics_csv_path = output_dir / 'training_metrics.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nTraining metrics saved to: {metrics_csv_path}")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
