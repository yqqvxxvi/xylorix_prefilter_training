#!/usr/bin/env python3
"""
Full classification pipeline: Wood vs Non-Wood -> VoL Filtering -> Usability Classification

This script orchestrates the complete pipeline for wood microscopy image classification:
1. Stage 1: Classify all images as wood vs non-wood
2. Stage 2: Filter wood images by VoL (blur detection)
3. Stage 3: Classify clear wood images as usable vs unusable

Usage:
    # Using CNN models for both stages
    python scripts/pipeline_full_classification.py \
        --input-dir dataset/raw_images/ \
        --output-dir outputs/pipeline_results/ \
        --wood-model models/wood/best_model.pt \
        --wood-model-type cnn \
        --usability-model models/usability/best_model.pt \
        --usability-model-type cnn

    # Using mixed models (RF for wood, CNN for usability)
    python scripts/pipeline_full_classification.py \
        --input-dir dataset/raw_images/ \
        --output-dir outputs/pipeline_results/ \
        --wood-model models/wood/blob_ml/ \
        --wood-model-type rf \
        --use-texture \
        --usability-model models/usability/best_model.pt \
        --usability-model-type cnn \
        --vol-threshold 900

    # With confidence thresholds
    python scripts/pipeline_full_classification.py \
        --input-dir dataset/raw_images/ \
        --output-dir outputs/pipeline_results/ \
        --wood-model models/wood/best_model.pt \
        --wood-model-type cnn \
        --usability-model models/usability/best_model.pt \
        --usability-model-type cnn \
        --wood-confidence-threshold 0.8 \
        --usability-confidence-threshold 0.7
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import cv2
from tqdm import tqdm
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import WoodFeatureExtractor
from src.features.quality import compute_variance_of_laplacian
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
        model = ResNet18(num_classes=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model, None, get_test_transforms()

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_with_features(image_path, model, feature_extractor, model_type, class_names):
    """Predict using feature-based model"""
    try:
        features = feature_extractor.extract(image_path)

        if features['is_clear'] == 0:
            return {
                'prediction': 'blurry',
                'confidence': 0.0,
                'probability': 0.0,
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
                prob_positive = model_obj(X_tensor).item()
            pred = 1 if prob_positive > 0.5 else 0
            prob = [1 - prob_positive, prob_positive]

        label = class_names[pred]
        confidence = prob[pred]

        return {
            'prediction': label,
            'confidence': confidence,
            'probability': prob[1],
            'vol_score': features['vol_score'],
            'is_clear': True
        }
    except Exception as e:
        return {
            'prediction': 'error',
            'confidence': 0.0,
            'probability': 0.0,
            'vol_score': 0.0,
            'is_clear': False,
            'error': str(e)
        }


def predict_with_cnn(image_path, model, transform, device, class_names):
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
        label = class_names[pred]

        return {
            'prediction': label,
            'confidence': probs[pred],
            'probability': probs[1],
            'vol_score': None,
            'is_clear': True
        }
    except Exception as e:
        return {
            'prediction': 'error',
            'confidence': 0.0,
            'probability': 0.0,
            'vol_score': None,
            'is_clear': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Full pipeline: Wood Classification -> VoL Filtering -> Usability Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/output
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing raw images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for pipeline results')

    # Wood classification model
    parser.add_argument('--wood-model', type=str, required=True,
                       help='Path to wood classification model')
    parser.add_argument('--wood-model-type', type=str, required=True,
                       choices=['rf', 'mlp', 'cnn'],
                       help='Wood model type')

    # Usability classification model
    parser.add_argument('--usability-model', type=str, required=True,
                       help='Path to usability classification model')
    parser.add_argument('--usability-model-type', type=str, required=True,
                       choices=['rf', 'mlp', 'cnn'],
                       help='Usability model type')

    # VoL filtering
    parser.add_argument('--vol-threshold', type=float, default=900,
                       help='VoL threshold for blur detection (default: 900)')
    parser.add_argument('--skip-vol-filtering', action='store_true',
                       help='Skip VoL filtering stage (not recommended)')

    # Feature extraction (for RF/MLP models)
    parser.add_argument('--use-texture', action='store_true',
                       help='Use texture features (for RF/MLP models)')

    # Confidence thresholds
    parser.add_argument('--wood-confidence-threshold', type=float, default=0.0,
                       help='Minimum confidence for wood classification (default: 0.0)')
    parser.add_argument('--usability-confidence-threshold', type=float, default=0.0,
                       help='Minimum confidence for usability classification (default: 0.0)')

    # Hardware
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto-detect)')

    # Options
    parser.add_argument('--copy-files', action='store_true',
                       help='Copy images to organized subdirectories')
    parser.add_argument('--save-intermediates', action='store_true',
                       help='Save intermediate results for each stage')

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Create output structure
    stage1_dir = output_dir / 'stage1_wood_classification'
    stage2_dir = output_dir / 'stage2_vol_filtering'
    stage3_dir = output_dir / 'stage3_usability_classification'
    final_dir = output_dir / 'final_results'

    for d in [stage1_dir, stage2_dir, stage3_dir, final_dir]:
        d.mkdir(exist_ok=True)

    if args.copy_files:
        (final_dir / 'usable').mkdir(exist_ok=True)
        (final_dir / 'unusable').mkdir(exist_ok=True)
        (final_dir / 'non_wood').mkdir(exist_ok=True)
        (final_dir / 'blurry').mkdir(exist_ok=True)
        (final_dir / 'low_confidence').mkdir(exist_ok=True)
        (final_dir / 'error').mkdir(exist_ok=True)

    # Print pipeline configuration
    print("=" * 80)
    print("FULL CLASSIFICATION PIPELINE")
    print("=" * 80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"\nPipeline Stages:")
    print(f"  1. Wood Classification ({args.wood_model_type.upper()})")
    print(f"  2. VoL Filtering (threshold={args.vol_threshold})" + (" [SKIPPED]" if args.skip_vol_filtering else ""))
    print(f"  3. Usability Classification ({args.usability_model_type.upper()})")
    print(f"\nConfidence Thresholds:")
    print(f"  Wood: {args.wood_confidence_threshold}")
    print(f"  Usability: {args.usability_confidence_threshold}")
    print("=" * 80)

    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(input_dir.glob(ext)))

    print(f"\nFound {len(image_files)} images")

    # ========== STAGE 1: WOOD vs NON-WOOD CLASSIFICATION ==========
    print("\n" + "=" * 80)
    print("STAGE 1: WOOD vs NON-WOOD CLASSIFICATION")
    print("=" * 80)
    print(f"Loading wood classification model from: {args.wood_model}")

    # Load wood model
    if args.wood_model_type in ['rf', 'mlp']:
        wood_model, wood_feature_extractor, wood_transform = load_model(
            args.wood_model_type, args.wood_model, device, args.use_texture, args.vol_threshold
        )
    else:
        wood_model, wood_feature_extractor, wood_transform = load_model(
            args.wood_model_type, args.wood_model, device
        )

    # Process images
    print("\nClassifying images...")
    stage1_results = []
    for img_path in tqdm(image_files, desc="Stage 1"):
        if args.wood_model_type in ['rf', 'mlp']:
            result = predict_with_features(
                img_path, wood_model, wood_feature_extractor, args.wood_model_type,
                class_names=['non_wood', 'wood']
            )
        else:
            result = predict_with_cnn(
                img_path, wood_model, wood_transform, device,
                class_names=['non_wood', 'wood']
            )

        result['filename'] = img_path.name
        result['filepath'] = str(img_path)
        result['stage'] = 'wood_classification'
        stage1_results.append(result)

    # Save stage 1 results
    df_stage1 = pd.DataFrame(stage1_results)
    if args.save_intermediates:
        df_stage1.to_csv(stage1_dir / 'results.csv', index=False)

    # Summary
    wood_images = df_stage1[
        (df_stage1['prediction'] == 'wood') &
        (df_stage1['confidence'] >= args.wood_confidence_threshold)
    ]
    print(f"\nStage 1 Results:")
    print(f"  Total images: {len(df_stage1)}")
    print(f"  Wood: {len(wood_images)} (passing confidence threshold)")
    print(f"  Non-wood: {len(df_stage1) - len(wood_images)}")

    # ========== STAGE 2: VOL FILTERING ==========
    if not args.skip_vol_filtering:
        print("\n" + "=" * 80)
        print("STAGE 2: VOL FILTERING (Blur Detection)")
        print("=" * 80)

        print(f"Filtering {len(wood_images)} wood images by VoL (threshold={args.vol_threshold})...")

        stage2_results = []
        for _, row in tqdm(wood_images.iterrows(), total=len(wood_images), desc="Stage 2"):
            img_path = Path(row['filepath'])

            try:
                # Compute VoL if not already computed
                if pd.isna(row.get('vol_score')) or row.get('vol_score') is None:
                    vol_score = compute_variance_of_laplacian(str(img_path))
                else:
                    vol_score = row['vol_score']

                is_clear = vol_score >= args.vol_threshold

                result = {
                    'filename': row['filename'],
                    'filepath': str(img_path),
                    'vol_score': vol_score,
                    'is_clear': is_clear,
                    'status': 'clear' if is_clear else 'blurry',
                    'wood_confidence': row['confidence']
                }
            except Exception as e:
                result = {
                    'filename': row['filename'],
                    'filepath': str(img_path),
                    'vol_score': 0.0,
                    'is_clear': False,
                    'status': 'error',
                    'error': str(e),
                    'wood_confidence': row['confidence']
                }

            stage2_results.append(result)

        # Save stage 2 results
        df_stage2 = pd.DataFrame(stage2_results)
        if args.save_intermediates:
            df_stage2.to_csv(stage2_dir / 'results.csv', index=False)

        # Filter clear images
        clear_wood_images = df_stage2[df_stage2['is_clear'] == True]

        print(f"\nStage 2 Results:")
        print(f"  Clear wood images: {len(clear_wood_images)}")
        print(f"  Blurry wood images: {len(df_stage2) - len(clear_wood_images)}")
    else:
        print("\n[Skipping VoL filtering as requested]")
        clear_wood_images = wood_images
        df_stage2 = None

    # ========== STAGE 3: USABILITY CLASSIFICATION ==========
    print("\n" + "=" * 80)
    print("STAGE 3: USABILITY CLASSIFICATION")
    print("=" * 80)
    print(f"Loading usability classification model from: {args.usability_model}")

    # Load usability model
    if args.usability_model_type in ['rf', 'mlp']:
        usability_model, usability_feature_extractor, usability_transform = load_model(
            args.usability_model_type, args.usability_model, device, args.use_texture, args.vol_threshold
        )
    else:
        usability_model, usability_feature_extractor, usability_transform = load_model(
            args.usability_model_type, args.usability_model, device
        )

    # Process clear wood images
    print(f"\nClassifying {len(clear_wood_images)} clear wood images...")
    stage3_results = []

    for idx, row in tqdm(clear_wood_images.iterrows(), total=len(clear_wood_images), desc="Stage 3"):
        img_path = Path(row['filepath'])

        if args.usability_model_type in ['rf', 'mlp']:
            result = predict_with_features(
                img_path, usability_model, usability_feature_extractor, args.usability_model_type,
                class_names=['unusable', 'usable']
            )
        else:
            result = predict_with_cnn(
                img_path, usability_model, usability_transform, device,
                class_names=['unusable', 'usable']
            )

        result['filename'] = row['filename']
        result['filepath'] = str(img_path)
        result['wood_confidence'] = row.get('confidence', row.get('wood_confidence', None))
        if df_stage2 is not None:
            result['vol_score'] = row.get('vol_score', None)
        stage3_results.append(result)

    # Save stage 3 results
    df_stage3 = pd.DataFrame(stage3_results)
    if args.save_intermediates:
        df_stage3.to_csv(stage3_dir / 'results.csv', index=False)

    # Filter by confidence threshold
    high_conf_usable = df_stage3[
        (df_stage3['prediction'] == 'usable') &
        (df_stage3['confidence'] >= args.usability_confidence_threshold)
    ]
    high_conf_unusable = df_stage3[
        (df_stage3['prediction'] == 'unusable') &
        (df_stage3['confidence'] >= args.usability_confidence_threshold)
    ]

    print(f"\nStage 3 Results:")
    print(f"  Total clear wood images: {len(df_stage3)}")
    print(f"  Usable (high confidence): {len(high_conf_usable)}")
    print(f"  Unusable (high confidence): {len(high_conf_unusable)}")
    print(f"  Low confidence: {len(df_stage3) - len(high_conf_usable) - len(high_conf_unusable)}")

    # ========== FINAL RESULTS ==========
    print("\n" + "=" * 80)
    print("FINAL PIPELINE RESULTS")
    print("=" * 80)

    # Combine all results
    final_results = []

    # Add usability classified images
    for _, row in df_stage3.iterrows():
        category = row['prediction'] if row['confidence'] >= args.usability_confidence_threshold else 'low_confidence'
        final_results.append({
            'filename': row['filename'],
            'filepath': row['filepath'],
            'final_category': category,
            'wood_confidence': row['wood_confidence'],
            'usability_prediction': row['prediction'],
            'usability_confidence': row['confidence'],
            'vol_score': row.get('vol_score', None)
        })

        # Copy files if requested
        if args.copy_files:
            dest = final_dir / category / row['filename']
            shutil.copy2(row['filepath'], dest)

    # Add blurry wood images
    if df_stage2 is not None:
        blurry_wood = df_stage2[df_stage2['is_clear'] == False]
        for _, row in blurry_wood.iterrows():
            final_results.append({
                'filename': row['filename'],
                'filepath': row['filepath'],
                'final_category': 'blurry',
                'wood_confidence': row['wood_confidence'],
                'usability_prediction': None,
                'usability_confidence': None,
                'vol_score': row['vol_score']
            })

            if args.copy_files:
                dest = final_dir / 'blurry' / row['filename']
                shutil.copy2(row['filepath'], dest)

    # Add non-wood images
    non_wood = df_stage1[
        (df_stage1['prediction'] != 'wood') |
        (df_stage1['confidence'] < args.wood_confidence_threshold)
    ]
    for _, row in non_wood.iterrows():
        final_results.append({
            'filename': row['filename'],
            'filepath': row['filepath'],
            'final_category': 'non_wood',
            'wood_confidence': row['confidence'],
            'usability_prediction': None,
            'usability_confidence': None,
            'vol_score': row.get('vol_score', None)
        })

        if args.copy_files:
            dest = final_dir / 'non_wood' / row['filename']
            shutil.copy2(row['filepath'], dest)

    # Save final results
    df_final = pd.DataFrame(final_results)
    df_final.to_csv(final_dir / 'final_results.csv', index=False)

    # Print final summary
    print(f"\nTotal images processed: {len(image_files)}")
    print(f"\nFinal distribution:")
    print(df_final['final_category'].value_counts().to_string())

    print(f"\n\nResults saved to:")
    print(f"  Final CSV: {final_dir / 'final_results.csv'}")
    if args.save_intermediates:
        print(f"  Stage 1: {stage1_dir / 'results.csv'}")
        print(f"  Stage 2: {stage2_dir / 'results.csv'}")
        print(f"  Stage 3: {stage3_dir / 'results.csv'}")
    if args.copy_files:
        print(f"  Organized files: {final_dir}")

    print("=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
