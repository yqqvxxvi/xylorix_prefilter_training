#!/usr/bin/env python3
"""
Full Pipeline Testing Script with TFLite Models
Pipeline: Wood Classification -> Sobel Spectrum -> Usability Classification

All images are processed through all 3 stages with results saved at each step.

Usage:
    python scripts/pipeline_tflite_full_test.py \
        --input-dir /path/to/images \
        --output-dir /path/to/results \
        --wood-model /path/to/wood_model.tflite \
        --usability-model /path/to/usability_model.tflite

Pipeline Stages:
1. Wood Classification (TFLite): Classify as wood vs non-wood
2. Sobel Spectrum Assessment: Assess image sharpness (clear/borderline/very_blurry/blurry)
3. Usability Classification (TFLite): Classify clear images as usable vs unusable

All images proceed through all stages regardless of earlier classifications.
"""

import argparse
import sys
from pathlib import Path
import shutil
from typing import Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
PREFILTERED_PATH = Path("/Users/youqing/Documents/prefiltered_pipeline_proposal")
sys.path.insert(0, str(PREFILTERED_PATH))

from utils.quality import calculate_sobel_variance

# Import TFLite
try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow is not installed. Please install with: pip install tensorflow")
    sys.exit(1)

# Import sklearn for confusion matrix
try:
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Warning: scikit-learn not installed. Confusion matrix features will be limited.")
    print("   Install with: pip install scikit-learn seaborn")


class TFLiteFullPipeline:
    """Full pipeline processor using TFLite models for wood and usability classification"""

    # Sobel spectrum thresholds
    CLEAR_THRESHOLD = 3788
    BORDERLINE_THRESHOLD = 2500
    VERY_BLURRY_THRESHOLD = 1000

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        wood_model_path: Path,
        usability_model_path: Path,
        wood_threshold: float = 0.5,
        usability_threshold: float = 0.5,
        labeled_input: bool = False,
        wood_labels: list = None,
        usability_labels: list = None,
        sobel_labels: list = None
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.wood_model_path = Path(wood_model_path)
        self.usability_model_path = Path(usability_model_path)
        self.wood_threshold = wood_threshold
        self.usability_threshold = usability_threshold
        self.labeled_input = labeled_input
        self.wood_labels = wood_labels or ['non_wood', 'wood']
        self.usability_labels = usability_labels or ['unusable', 'usable']
        self.sobel_labels = sobel_labels or ['clear', 'borderline', 'very_blurry', 'blurry']

        # Create output directories
        self.stage1_dir = self.output_dir / "stage1_wood_classification"
        self.stage2_dir = self.output_dir / "stage2_sobel_spectrum"
        self.stage3_dir = self.output_dir / "stage3_usability"
        self.viz_dir = self.output_dir / "visualizations"

        # Stage 1: Wood classification
        (self.stage1_dir / "wood").mkdir(parents=True, exist_ok=True)
        (self.stage1_dir / "non_wood").mkdir(parents=True, exist_ok=True)

        # Stage 2: Sobel spectrum (4 categories)
        (self.stage2_dir / "clear").mkdir(parents=True, exist_ok=True)
        (self.stage2_dir / "borderline").mkdir(parents=True, exist_ok=True)
        (self.stage2_dir / "very_blurry").mkdir(parents=True, exist_ok=True)
        (self.stage2_dir / "blurry").mkdir(parents=True, exist_ok=True)

        # Stage 3: Usability
        (self.stage3_dir / "usable").mkdir(parents=True, exist_ok=True)
        (self.stage3_dir / "unusable").mkdir(parents=True, exist_ok=True)

        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Results storage - now per image with all stages
        self.per_image_results = {}  # filename -> {wood: {}, sobel: {}, usability: {}}

        # Load TFLite models
        print(f"Loading wood classification TFLite model from {self.wood_model_path}...")
        self.wood_interpreter = tf.lite.Interpreter(model_path=str(self.wood_model_path))
        self.wood_interpreter.allocate_tensors()
        self.wood_input_details = self.wood_interpreter.get_input_details()
        self.wood_output_details = self.wood_interpreter.get_output_details()

        print(f"Loading usability TFLite model from {self.usability_model_path}...")
        self.usability_interpreter = tf.lite.Interpreter(model_path=str(self.usability_model_path))
        self.usability_interpreter.allocate_tensors()
        self.usability_input_details = self.usability_interpreter.get_input_details()
        self.usability_output_details = self.usability_interpreter.get_output_details()

        # Print model input shapes
        print(f"\nWood model input shape: {self.wood_input_details[0]['shape']}")
        print(f"Usability model input shape: {self.usability_input_details[0]['shape']}")
        print("Models loaded successfully!")

    def _classify_sobel_category(self, sobel_score: float) -> str:
        """Classify image based on Sobel variance spectrum"""
        if sobel_score > self.CLEAR_THRESHOLD:
            return "clear"
        elif sobel_score > self.BORDERLINE_THRESHOLD:
            return "borderline"
        elif sobel_score > self.VERY_BLURRY_THRESHOLD:
            return "very_blurry"
        else:
            return "blurry"

    def _classify_tflite(
        self,
        img_path: Path,
        interpreter,
        input_details,
        output_details,
        threshold: float
    ) -> Tuple[int, float]:
        """Classify image using TFLite model"""
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Get expected number of channels from model
        expected_shape = input_details[0]['shape']
        expected_channels = expected_shape[3] if len(expected_shape) == 4 else 3

        # Convert to grayscale if model expects 1 channel, otherwise keep RGB
        if expected_channels == 1:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # Convert BGR to RGB for 3-channel models
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model input size (224x224)
        img = cv2.resize(img, (224, 224))

        # Normalize and prepare input
        img_array = img.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Add channel dimension only if grayscale
        if expected_channels == 1 and len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=-1)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])

        # Apply sigmoid to get probability
        probability = 1 / (1 + np.exp(-output[0][0]))

        # Binary classification
        prediction = 1 if probability >= threshold else 0

        return prediction, float(probability)

    def process_batch(self):
        """Process all images through the pipeline"""
        # Get all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        if self.labeled_input:
            # For labeled input, expect structure: input_dir/class_name/image.jpg
            image_files = []
            true_labels = {}  # filename -> {wood: label, usability: label, sobel: label}

            print(f"\n{'='*70}")
            print("Processing LABELED input directory")
            print(f"{'='*70}")

            # Collect images from labeled subdirectories
            for class_dir in self.input_dir.iterdir():
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                print(f"Found class directory: {class_name}")

                for img_path in class_dir.rglob('*'):
                    if img_path.suffix.lower() in image_extensions:
                        image_files.append(img_path)

                        # Store true labels - using directory name as ground truth
                        # Assume directory structure can indicate wood/usability/sobel labels
                        true_labels[img_path.name] = {
                            'wood': class_name if class_name in self.wood_labels else None,
                            'usability': class_name if class_name in self.usability_labels else None,
                            'sobel': class_name if class_name in self.sobel_labels else None,
                            'class_dir': class_name
                        }
        else:
            # Unlabeled input
            image_files = [
                f for f in self.input_dir.rglob('*')
                if f.suffix.lower() in image_extensions
            ]
            true_labels = None

        print(f"\n{'='*70}")
        print(f"Found {len(image_files)} images to process")
        print(f"{'='*70}")

        # Initialize results for all images
        for img_path in image_files:
            self.per_image_results[img_path.name] = {
                'filename': img_path.name,
                'filepath': str(img_path),
                'wood_classification': {},
                'sobel_spectrum': {},
                'usability_classification': {},
                'true_labels': true_labels[img_path.name] if true_labels else {}
            }

        # ===== STAGE 1: Wood Classification =====
        print(f"\n{'='*70}")
        print("STAGE 1: Wood Classification (TFLite)")
        print(f"{'='*70}")

        wood_count = 0
        for img_path in tqdm(image_files, desc="Classifying wood"):
            try:
                prediction, confidence = self._classify_tflite(
                    img_path,
                    self.wood_interpreter,
                    self.wood_input_details,
                    self.wood_output_details,
                    self.wood_threshold
                )

                # Store results (REVERSED: 1=non_wood, 0=wood)
                self.per_image_results[img_path.name]['wood_classification'] = {
                    'prediction': 'non_wood' if prediction == 1 else 'wood',
                    'prediction_value': int(prediction),
                    'confidence': float(confidence),
                    'threshold': self.wood_threshold
                }

                if prediction == 0:  # Changed from prediction == 1
                    wood_count += 1

                # Copy to folder (REVERSED)
                dest_folder = self.stage1_dir / ("non_wood" if prediction == 1 else "wood")
                shutil.copy2(img_path, dest_folder / img_path.name)

            except Exception as e:
                print(f"\nError processing {img_path.name}: {e}")
                self.per_image_results[img_path.name]['wood_classification'] = {
                    'prediction': 'error',
                    'prediction_value': 0,
                    'confidence': 0.0,
                    'threshold': self.wood_threshold,
                    'error': str(e)
                }

        non_wood_count = len(image_files) - wood_count

        print(f"\nWood Classification Results:")
        print(f"  Wood:     {wood_count} ({wood_count/len(image_files)*100:.1f}%)")
        print(f"  Non-wood: {non_wood_count} ({non_wood_count/len(image_files)*100:.1f}%)")
        print(f"\n→ Continuing with ALL {len(image_files)} images to Stage 2...")

        # ===== STAGE 2: Sobel Spectrum Assessment =====
        print(f"\n{'='*70}")
        print("STAGE 2: Sobel Spectrum Assessment")
        print(f"{'='*70}")
        print(f"Thresholds:")
        print(f"  Clear:        > {self.CLEAR_THRESHOLD}")
        print(f"  Borderline:   {self.BORDERLINE_THRESHOLD} - {self.CLEAR_THRESHOLD}")
        print(f"  Very Blurry:  {self.VERY_BLURRY_THRESHOLD} - {self.BORDERLINE_THRESHOLD}")
        print(f"  Blurry:       0 - {self.VERY_BLURRY_THRESHOLD}")
        print(f"{'='*70}")

        category_counts = {"clear": 0, "borderline": 0, "very_blurry": 0, "blurry": 0}

        for img_path in tqdm(image_files, desc="Analyzing Sobel spectrum"):
            try:
                sobel_score = calculate_sobel_variance(str(img_path), crop=True)
                category = self._classify_sobel_category(sobel_score)

                # Store results
                self.per_image_results[img_path.name]['sobel_spectrum'] = {
                    'sobel_score': float(sobel_score),
                    'category': category,
                    'thresholds': {
                        'clear': self.CLEAR_THRESHOLD,
                        'borderline': self.BORDERLINE_THRESHOLD,
                        'very_blurry': self.VERY_BLURRY_THRESHOLD
                    }
                }

                category_counts[category] += 1

                # Copy to appropriate folder
                dest_folder = self.stage2_dir / category
                shutil.copy2(img_path, dest_folder / img_path.name)

            except Exception as e:
                print(f"\nError processing {img_path.name}: {e}")
                self.per_image_results[img_path.name]['sobel_spectrum'] = {
                    'sobel_score': 0.0,
                    'category': 'error',
                    'error': str(e)
                }

        print(f"\nSobel Spectrum Results (from {len(image_files)} images):")
        print(f"  Clear (>{self.CLEAR_THRESHOLD}):           "
              f"{category_counts['clear']} ({category_counts['clear']/len(image_files)*100:.1f}%)")
        print(f"  Borderline ({self.BORDERLINE_THRESHOLD}-{self.CLEAR_THRESHOLD}):  "
              f"{category_counts['borderline']} ({category_counts['borderline']/len(image_files)*100:.1f}%)")
        print(f"  Very Blurry ({self.VERY_BLURRY_THRESHOLD}-{self.BORDERLINE_THRESHOLD}): "
              f"{category_counts['very_blurry']} ({category_counts['very_blurry']/len(image_files)*100:.1f}%)")
        print(f"  Blurry (0-{self.VERY_BLURRY_THRESHOLD}):          "
              f"{category_counts['blurry']} ({category_counts['blurry']/len(image_files)*100:.1f}%)")
        print(f"\n→ Continuing with ALL {len(image_files)} images to Stage 3...")

        # ===== STAGE 3: Usability Classification =====
        print(f"\n{'='*70}")
        print("STAGE 3: Usability Classification (TFLite - ALL Images)")
        print(f"{'='*70}")

        usable_count = 0
        for img_path in tqdm(image_files, desc="Classifying usability"):
            try:
                prediction, confidence = self._classify_tflite(
                    img_path,
                    self.usability_interpreter,
                    self.usability_input_details,
                    self.usability_output_details,
                    self.usability_threshold
                )

                # Store results
                self.per_image_results[img_path.name]['usability_classification'] = {
                    'prediction': 'usable' if prediction == 1 else 'unusable',
                    'prediction_value': int(prediction),
                    'confidence': float(confidence),
                    'threshold': self.usability_threshold
                }

                if prediction == 1:
                    usable_count += 1

                # Copy to folder
                dest_folder = self.stage3_dir / ("usable" if prediction == 1 else "unusable")
                shutil.copy2(img_path, dest_folder / img_path.name)

            except Exception as e:
                print(f"\nError processing {img_path.name}: {e}")
                self.per_image_results[img_path.name]['usability_classification'] = {
                    'prediction': 'error',
                    'prediction_value': 0,
                    'confidence': 0.0,
                    'threshold': self.usability_threshold,
                    'error': str(e)
                }

        unusable_count = len(image_files) - usable_count

        print(f"\nUsability Results (from {len(image_files)} images):")
        print(f"  Usable:   {usable_count} ({usable_count/len(image_files)*100:.1f}%)")
        print(f"  Unusable: {unusable_count} ({unusable_count/len(image_files)*100:.1f}%)")

        # Save results
        self._save_results_json()

        # Generate confusion matrices if labeled input
        if self.labeled_input:
            self._generate_confusion_matrices()

        # Print final summary
        print(f"\n{'='*70}")
        print("PIPELINE SUMMARY")
        print(f"{'='*70}")
        print(f"Total images processed:    {len(image_files)}")
        print(f"\nStage 1 - Wood Classification:")
        print(f"  Wood:                    {wood_count} ({wood_count/len(image_files)*100:.1f}%)")
        print(f"  Non-wood:                {non_wood_count} ({non_wood_count/len(image_files)*100:.1f}%)")
        print(f"\nStage 2 - Sobel Spectrum:")
        print(f"  Clear (>3788):           {category_counts['clear']} ({category_counts['clear']/len(image_files)*100:.1f}%)")
        print(f"  Borderline (2500-3788):  {category_counts['borderline']} ({category_counts['borderline']/len(image_files)*100:.1f}%)")
        print(f"  Very Blurry (1000-2500): {category_counts['very_blurry']} ({category_counts['very_blurry']/len(image_files)*100:.1f}%)")
        print(f"  Blurry (0-1000):         {category_counts['blurry']} ({category_counts['blurry']/len(image_files)*100:.1f}%)")
        print(f"\nStage 3 - Usability:")
        print(f"  Usable:                  {usable_count} ({usable_count/len(image_files)*100:.1f}%)")
        print(f"  Unusable:                {unusable_count} ({unusable_count/len(image_files)*100:.1f}%)")
        print(f"{'='*70}")

    def _save_results_json(self):
        """Save detailed results to JSON files"""

        # Calculate summary statistics
        total_images = len(self.per_image_results)

        wood_images = sum(1 for r in self.per_image_results.values()
                         if r['wood_classification'].get('prediction') == 'wood')

        category_counts = {"clear": 0, "borderline": 0, "very_blurry": 0, "blurry": 0}
        for r in self.per_image_results.values():
            cat = r['sobel_spectrum'].get('category')
            if cat in category_counts:
                category_counts[cat] += 1

        usable_count = sum(1 for r in self.per_image_results.values()
                          if r['usability_classification'].get('prediction') == 'usable')

        # Create summary JSON
        summary = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'input_directory': str(self.input_dir),
                'model_type': 'TFLite',
                'total_images': total_images
            },
            'thresholds': {
                'wood_threshold': self.wood_threshold,
                'usability_threshold': self.usability_threshold,
                'sobel_clear': f'> {self.CLEAR_THRESHOLD}',
                'sobel_borderline': f'{self.BORDERLINE_THRESHOLD} - {self.CLEAR_THRESHOLD}',
                'sobel_very_blurry': f'{self.VERY_BLURRY_THRESHOLD} - {self.BORDERLINE_THRESHOLD}',
                'sobel_blurry': f'0 - {self.VERY_BLURRY_THRESHOLD}'
            },
            'statistics': {
                'stage1_wood_classification': {
                    'wood': wood_images,
                    'non_wood': total_images - wood_images,
                    'wood_percentage': wood_images / total_images * 100 if total_images > 0 else 0
                },
                'stage2_sobel_spectrum': {
                    'clear': category_counts['clear'],
                    'borderline': category_counts['borderline'],
                    'very_blurry': category_counts['very_blurry'],
                    'blurry': category_counts['blurry']
                },
                'stage3_usability': {
                    'usable': usable_count,
                    'unusable': total_images - usable_count,
                    'usability_percentage': usable_count / total_images * 100 if total_images > 0 else 0
                }
            }
        }

        # Save summary
        summary_file = self.output_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_file}")

        # Save detailed per-image results
        detailed_file = self.output_dir / "pipeline_detailed_results.json"
        with open(detailed_file, 'w') as f:
            json.dump({
                'metadata': summary['metadata'],
                'thresholds': summary['thresholds'],
                'per_image_results': list(self.per_image_results.values())
            }, f, indent=2)
        print(f"Detailed results saved to {detailed_file}")

    def _generate_confusion_matrices(self):
        """Generate confusion matrices for all pipeline stages"""
        if not self.labeled_input or not SKLEARN_AVAILABLE:
            if not self.labeled_input:
                print("\n⚠️  Skipping confusion matrices (unlabeled input)")
            else:
                print("\n⚠️  Skipping confusion matrices (scikit-learn not available)")
            return

        print(f"\n{'='*70}")
        print("GENERATING CONFUSION MATRICES")
        print(f"{'='*70}")

        # Prepare data for each stage
        stages = []

        # Stage 1: Wood Classification
        wood_y_true = []
        wood_y_pred = []
        for img_name, result in self.per_image_results.items():
            true_label = result.get('true_labels', {}).get('wood')
            if true_label and true_label in self.wood_labels:
                wood_y_true.append(self.wood_labels.index(true_label))
                pred_label = result['wood_classification'].get('prediction')
                if pred_label in self.wood_labels:
                    wood_y_pred.append(self.wood_labels.index(pred_label))
                else:
                    wood_y_pred.append(-1)  # Unknown

        if len(wood_y_true) > 0:
            stages.append({
                'name': 'Wood Classification',
                'y_true': wood_y_true,
                'y_pred': wood_y_pred,
                'labels': self.wood_labels,
                'filename': 'confusion_matrix_wood.png'
            })

        # Stage 2: Sobel Spectrum
        sobel_y_true = []
        sobel_y_pred = []
        for img_name, result in self.per_image_results.items():
            true_label = result.get('true_labels', {}).get('sobel')
            if true_label and true_label in self.sobel_labels:
                sobel_y_true.append(self.sobel_labels.index(true_label))
                pred_label = result['sobel_spectrum'].get('category')
                if pred_label in self.sobel_labels:
                    sobel_y_pred.append(self.sobel_labels.index(pred_label))
                else:
                    sobel_y_pred.append(-1)

        if len(sobel_y_true) > 0:
            stages.append({
                'name': 'Sobel Spectrum',
                'y_true': sobel_y_true,
                'y_pred': sobel_y_pred,
                'labels': self.sobel_labels,
                'filename': 'confusion_matrix_sobel.png'
            })

        # Stage 3: Usability Classification
        usability_y_true = []
        usability_y_pred = []
        for img_name, result in self.per_image_results.items():
            true_label = result.get('true_labels', {}).get('usability')
            if true_label and true_label in self.usability_labels:
                usability_y_true.append(self.usability_labels.index(true_label))
                pred_label = result['usability_classification'].get('prediction')
                if pred_label in self.usability_labels:
                    usability_y_pred.append(self.usability_labels.index(pred_label))
                else:
                    usability_y_pred.append(-1)

        if len(usability_y_true) > 0:
            stages.append({
                'name': 'Usability Classification',
                'y_true': usability_y_true,
                'y_pred': usability_y_pred,
                'labels': self.usability_labels,
                'filename': 'confusion_matrix_usability.png'
            })

        # Generate confusion matrices for each stage
        if not stages:
            print("\n⚠️  No labeled data found for confusion matrix generation")
            print("   Make sure input directory structure matches expected labels")
            return

        # Create figure with all confusion matrices
        num_stages = len(stages)
        fig, axes = plt.subplots(1, num_stages, figsize=(8 * num_stages, 6))
        if num_stages == 1:
            axes = [axes]

        for idx, stage_data in enumerate(stages):
            ax = axes[idx]

            y_true = stage_data['y_true']
            y_pred = stage_data['y_pred']
            labels = stage_data['labels']
            stage_name = stage_data['name']

            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred)

            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels,
                       ax=ax, cbar_kws={'label': 'Count'})
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title(f'{stage_name}\nAccuracy: {accuracy:.2%}', fontsize=14, fontweight='bold')

            # Print text version
            print(f"\n{stage_name}:")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Total samples: {len(y_true)}")
            print("\nConfusion Matrix (rows=true, cols=predicted):")
            print("        " + "  ".join(f"{label:>12}" for label in labels))
            for i, label in enumerate(labels):
                print(f"{label:>10}: " + "  ".join(f"{cm[i][j]:>12}" for j in range(len(labels))))

            # Print classification report
            print(f"\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=labels, digits=4))

        plt.tight_layout()

        # Save combined confusion matrix
        cm_combined_path = self.output_dir / "confusion_matrices_all_stages.png"
        plt.savefig(cm_combined_path, dpi=300, bbox_inches='tight')
        print(f"\nCombined confusion matrices saved to: {cm_combined_path}")

        plt.show()

        # Save individual confusion matrices
        for idx, stage_data in enumerate(stages):
            fig_individual = plt.figure(figsize=(10, 8))

            y_true = stage_data['y_true']
            y_pred = stage_data['y_pred']
            labels = stage_data['labels']
            stage_name = stage_data['name']
            filename = stage_data['filename']

            cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
            accuracy = accuracy_score(y_true, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels,
                       cbar_kws={'label': 'Count'})
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.title(f'{stage_name}\nAccuracy: {accuracy:.2%}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            cm_path = self.output_dir / filename
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Individual confusion matrix saved to: {cm_path}")

    def create_sample_visualization(self):
        """Create sample visualization with one image per category"""
        print(f"\n{'='*70}")
        print("GENERATING SAMPLE VISUALIZATION")
        print(f"{'='*70}")

        categories = [
            ('Wood', self.stage1_dir / 'wood'),
            ('Non-Wood', self.stage1_dir / 'non_wood'),
            ('Clear', self.stage2_dir / 'clear'),
            ('Borderline', self.stage2_dir / 'borderline'),
            ('Very Blurry', self.stage2_dir / 'very_blurry'),
            ('Blurry', self.stage2_dir / 'blurry'),
            ('Usable', self.stage3_dir / 'usable'),
            ('Unusable', self.stage3_dir / 'unusable')
        ]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Sample Images by Classification Category', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for idx, (cat_name, cat_dir) in enumerate(categories):
            ax = axes[idx]
            image_files = list(cat_dir.glob('*'))

            if image_files:
                # Get the first image
                img_path = image_files[0]
                img = cv2.imread(str(img_path))

                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img)

                    # Build title with info
                    title = f'{cat_name}\n'
                    title += f'({len(image_files)} images)\n'

                    # Add scores if available
                    filename = img_path.name
                    if filename in self.per_image_results:
                        sobel_info = self.per_image_results[filename].get('sobel_spectrum', {})
                        if 'sobel_score' in sobel_info:
                            score = sobel_info['sobel_score']
                            title += f'Sobel: {score:.0f}'

                    ax.set_title(title, fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'{cat_name}\n(No images)',
                       ha='center', va='center', fontsize=10)

            ax.axis('off')

        plt.tight_layout()
        plt.show()

        print("✓ Displayed sample images grid")


def main():
    parser = argparse.ArgumentParser(
        description='Full Pipeline Test: Wood -> Sobel -> Usability (TFLite)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: ALL images are processed through ALL 3 stages. No filtering occurs.

Examples:
  # Basic usage
  python scripts/pipeline_tflite_full_test.py \\
      --input-dir ./my_images \\
      --output-dir ./results \\
      --wood-model outputs/tflite/wood/model_float32.tflite \\
      --usability-model outputs/tflite/usability/model_float32.tflite

  # With custom thresholds
  python scripts/pipeline_tflite_full_test.py \\
      --input-dir ./my_images \\
      --output-dir ./results \\
      --wood-model outputs/tflite/wood/model_float32.tflite \\
      --usability-model outputs/tflite/usability/model_float32.tflite \\
      --wood-threshold 0.7 \\
      --usability-threshold 0.6

Pipeline Stages:
  1. Wood Classification: Classify as wood vs non-wood
  2. Sobel Spectrum: Assess sharpness (clear/borderline/very_blurry/blurry)
  3. Usability: Classify as usable vs unusable

All images proceed through all stages regardless of earlier classifications.

Labeled Input:
  Use --labeled-input flag when your input directory has labeled subdirectories:

  # Example structure for wood classification testing:
  input_dir/
    wood/
      img1.jpg
      img2.jpg
    non_wood/
      img3.jpg
      img4.jpg

  # For labeled input, confusion matrices will be generated automatically

Output JSON Files:
  - pipeline_summary.json: Overall statistics and counts
  - pipeline_detailed_results.json: Per-image results with all confidences
  - confusion_matrices_all_stages.png: Combined confusion matrices (if labeled)
  - confusion_matrix_wood.png: Wood classification confusion matrix (if labeled)
  - confusion_matrix_usability.png: Usability confusion matrix (if labeled)
        """
    )
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing input images (can be labeled or unlabeled)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save all results, images, and visualizations')
    parser.add_argument('--wood-model', type=str, required=True,
                       help='Path to wood classification TFLite model')
    parser.add_argument('--usability-model', type=str, required=True,
                       help='Path to usability TFLite model')
    parser.add_argument('--wood-threshold', type=float, default=0.5,
                       help='Confidence threshold for wood classification (default: 0.5)')
    parser.add_argument('--usability-threshold', type=float, default=0.5,
                       help='Confidence threshold for usability (default: 0.5)')
    parser.add_argument('--labeled-input', action='store_true',
                       help='Input directory contains labeled subdirectories (e.g., input_dir/wood/, input_dir/non_wood/)')
    parser.add_argument('--wood-labels', type=str, nargs='+',
                       help='Wood classification labels (default: non_wood wood)')
    parser.add_argument('--usability-labels', type=str, nargs='+',
                       help='Usability labels (default: unusable usable)')
    parser.add_argument('--sobel-labels', type=str, nargs='+',
                       help='Sobel spectrum labels (default: clear borderline very_blurry blurry)')
    parser.add_argument('--show-viz', action='store_true',
                       help='Show sample visualization at the end')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("FULL PIPELINE TEST (TFLite)")
    print(f"{'='*70}")
    print(f"Input directory:     {args.input_dir}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Wood model:          {args.wood_model}")
    print(f"Usability model:     {args.usability_model}")
    print(f"Wood threshold:      {args.wood_threshold}")
    print(f"Usability threshold: {args.usability_threshold}")
    print(f"{'='*70}\n")

    # Create processor
    processor = TFLiteFullPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        wood_model_path=args.wood_model,
        usability_model_path=args.usability_model,
        wood_threshold=args.wood_threshold,
        usability_threshold=args.usability_threshold,
        labeled_input=args.labeled_input,
        wood_labels=args.wood_labels,
        usability_labels=args.usability_labels,
        sobel_labels=args.sobel_labels
    )

    # Process batch
    processor.process_batch()

    # Create visualizations if requested
    if args.show_viz:
        processor.create_sample_visualization()

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"\nNOTE: ALL images were processed through ALL 3 stages.")
    print(f"      Images are categorized and saved at each stage.")
    if args.labeled_input:
        print(f"\nLABELED INPUT: Confusion matrices generated for evaluation.")
    print(f"\nOutput Structure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── stage1_wood_classification/")
    print(f"  │   ├── wood/               (classified as wood)")
    print(f"  │   └── non_wood/           (classified as non-wood)")
    print(f"  ├── stage2_sobel_spectrum/")
    print(f"  │   ├── clear/              (Sobel > 3788)")
    print(f"  │   ├── borderline/         (Sobel 2500-3788)")
    print(f"  │   ├── very_blurry/        (Sobel 1000-2500)")
    print(f"  │   └── blurry/             (Sobel 0-1000)")
    print(f"  ├── stage3_usability/")
    print(f"  │   ├── usable/             (classified as usable)")
    print(f"  │   └── unusable/           (classified as unusable)")
    print(f"  ├── pipeline_summary.json         (overall statistics)")
    print(f"  └── pipeline_detailed_results.json (per-image confidences)")
    if args.labeled_input:
        print(f"  ├── confusion_matrices_all_stages.png  (combined confusion matrices)")
        print(f"  ├── confusion_matrix_wood.png          (wood classification)")
        print(f"  └── confusion_matrix_usability.png     (usability classification)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
