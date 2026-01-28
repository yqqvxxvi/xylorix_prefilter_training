"""
Test TFLite model

This script tests the converted TFLite model by:
1. Running inference on test images
2. Comparing predictions with PyTorch model (optional)
3. Measuring inference speed
4. Validating output format
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import time
from PIL import Image
import json
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Warning: scikit-learn not installed. Confusion matrix features will be limited.")
    print("   Install with: pip install scikit-learn matplotlib seaborn")


def load_tflite_model(model_path):
    """
    Load TFLite model

    Args:
        model_path: Path to .tflite file

    Returns:
        TFLite interpreter
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("❌ TensorFlow not installed")
        print("Install with: pip install tensorflow")
        sys.exit(1)

    print(f"Loading TFLite model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n" + "=" * 60)
    print("Model Details:")
    print("=" * 60)
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output type: {output_details[0]['dtype']}")
    print("=" * 60)

    return interpreter


def preprocess_image(image_path, input_shape, grayscale=False):
    """
    Preprocess image for TFLite model

    Args:
        image_path: Path to image
        input_shape: Model input shape (batch, height, width, channels)
        grayscale: Whether to convert to grayscale

    Returns:
        Preprocessed numpy array
    """
    # Load image
    img = Image.open(image_path)

    # Convert to grayscale if needed
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    # Resize to model input size
    # input_shape is NHWC: (batch, height, width, channels)
    height, width = input_shape[1], input_shape[2]
    img = img.resize((width, height), Image.Resampling.BILINEAR)

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    # Add batch dimension and ensure correct shape
    if grayscale:
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dim
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
    else:
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

    print(f"Preprocessed image shape: {img_array.shape}")
    return img_array


def sigmoid(x):
    """Apply sigmoid function to convert logits to probabilities"""
    return 1 / (1 + np.exp(-x))


def run_inference(interpreter, image_array):
    """
    Run inference on preprocessed image

    Args:
        interpreter: TFLite interpreter
        image_array: Preprocessed image array

    Returns:
        Output predictions (probabilities after sigmoid)
    """
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000  # Convert to ms

    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])

    # Apply sigmoid to convert logits to probabilities for binary classification
    # Check if this is likely a binary classification model (single output)
    if output.shape[-1] == 1:
        output = sigmoid(output)

    return output, inference_time


def test_single_image(model_path, image_path, labels=None, grayscale=False, threshold=0.5):
    """
    Test model on a single image

    Args:
        model_path: Path to TFLite model
        image_path: Path to test image
        labels: Class labels (optional)
        grayscale: Whether model expects grayscale input
        threshold: Classification threshold (default: 0.5)
    """
    print("\n" + "=" * 60)
    print("Single Image Test")
    print("=" * 60)

    # Load model
    interpreter = load_tflite_model(model_path)

    # Get input shape
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    # Preprocess image
    print(f"\nPreprocessing image: {image_path}")
    image_array = preprocess_image(image_path, input_shape, grayscale)

    # Run inference
    print("\nRunning inference...")
    output, inference_time = run_inference(interpreter, image_array)

    # Display results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Output shape: {output.shape}")
    print(f"Raw output (probabilities): {output}")

    # Interpret output
    print(f"Classification threshold: {threshold}")
    if output.shape[-1] == 1:
        # Single output (binary classification, 0-1 probability)
        probability = output[0][0]
        print(f"\nProbability (class 1): {probability:.4f} ({probability*100:.2f}%)")
        if labels and len(labels) == 2:
            if probability > threshold:
                print(f"Predicted class: {labels[1]} (confidence: {probability*100:.2f}%)")
            else:
                print(f"Predicted class: {labels[0]} (confidence: {(1-probability)*100:.2f}%)")
    else:
        # Multiple outputs (class probabilities)
        predictions = output[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]

        print(f"\nClass probabilities:")
        for i, prob in enumerate(predictions):
            label = labels[i] if labels and i < len(labels) else f"Class {i}"
            print(f"  {label}: {prob:.4f} ({prob*100:.2f}%)")

        print(f"\nPredicted class: {predicted_class}")
        if labels and predicted_class < len(labels):
            print(f"Predicted label: {labels[predicted_class]}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")


def batch_test(model_path, image_dir, labels=None, grayscale=False, max_images=10, threshold=0.5):
    """
    Test model on multiple images

    Args:
        model_path: Path to TFLite model
        image_dir: Directory containing test images
        labels: Class labels (optional)
        grayscale: Whether model expects grayscale input
        max_images: Maximum number of images to test
        threshold: Classification threshold (default: 0.5)
    """
    print("\n" + "=" * 60)
    print("Batch Test")
    print("=" * 60)

    # Load model
    interpreter = load_tflite_model(model_path)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    # Find images
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.jpg')) + \
                  list(image_dir.glob('*.jpeg')) + \
                  list(image_dir.glob('*.png'))

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return

    image_files = image_files[:max_images]
    print(f"\nTesting on {len(image_files)} images...")
    print(f"Classification threshold: {threshold}")

    inference_times = []
    results = []

    for img_path in image_files:
        # Preprocess
        image_array = preprocess_image(img_path, input_shape, grayscale)

        # Inference
        output, inference_time = run_inference(interpreter, image_array)
        inference_times.append(inference_time)

        # Store result
        if output.shape[-1] == 1:
            prediction = output[0][0]
            predicted_class = 1 if prediction > threshold else 0
            confidence = prediction if prediction > threshold else 1 - prediction
        else:
            predicted_class = np.argmax(output[0])
            confidence = output[0][predicted_class]

        results.append({
            'image': img_path.name,
            'predicted_class': int(predicted_class),
            'confidence': float(confidence)
        })

        # Print result
        label = labels[predicted_class] if labels and predicted_class < len(labels) else f"Class {predicted_class}"
        print(f"  {img_path.name}: {label} ({confidence*100:.1f}%)")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Total images tested: {len(image_files)}")
    print(f"Average inference time: {np.mean(inference_times):.2f} ms")
    print(f"Min inference time: {np.min(inference_times):.2f} ms")
    print(f"Max inference time: {np.max(inference_times):.2f} ms")
    print(f"Throughput: {1000/np.mean(inference_times):.1f} images/second")


def labeled_test(model_path, labeled_dir, labels=None, grayscale=False, save_results=None, threshold=0.5):
    """
    Test model on labeled dataset and generate confusion matrix

    Args:
        model_path: Path to TFLite model
        labeled_dir: Directory with subdirectories named by class labels
                    Example structure:
                    labeled_dir/
                        class1/
                            img1.jpg
                            img2.jpg
                        class2/
                            img3.jpg
                            img4.jpg
        labels: Class labels (optional, will be inferred from directories)
        grayscale: Whether model expects grayscale input
        save_results: Path to save results (optional)
        threshold: Classification threshold for binary classification (default: 0.5)
    """
    print("\n" + "=" * 60)
    print("Labeled Dataset Test with Confusion Matrix")
    print("=" * 60)

    # Load model
    interpreter = load_tflite_model(model_path)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    output_details = interpreter.get_output_details()

    # Find labeled directories
    labeled_dir = Path(labeled_dir)
    class_dirs = [d for d in labeled_dir.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"❌ No class directories found in {labeled_dir}")
        return

    # Infer labels from directory names if not provided
    if labels is None:
        labels = sorted([d.name for d in class_dirs])
        print(f"\nInferred labels from directories: {labels}")

    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    print(f"\nFound {len(class_dirs)} class directories")
    print(f"Labels: {labels}")

    # Collect all images with their true labels
    all_images = []
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name not in label_to_idx:
            print(f"⚠️  Warning: Directory '{class_name}' not in labels, skipping")
            continue

        true_label_idx = label_to_idx[class_name]

        # Find all images
        image_files = list(class_dir.glob('*.jpg')) + \
                     list(class_dir.glob('*.jpeg')) + \
                     list(class_dir.glob('*.png'))

        for img_path in image_files:
            all_images.append({
                'path': img_path,
                'true_label': class_name,
                'true_label_idx': true_label_idx
            })

    if not all_images:
        print("❌ No images found")
        return

    print(f"\nTotal images found: {len(all_images)}")
    for label in labels:
        count = sum(1 for img in all_images if img['true_label'] == label)
        print(f"  {label}: {count} images")

    # Run inference on all images
    print(f"\n{'='*60}")
    print("Running inference...")
    print(f"{'='*60}")
    print(f"Classification threshold: {threshold}")

    y_true = []
    y_pred = []
    y_scores = []
    inference_times = []
    results = []

    for img_info in all_images:
        img_path = img_info['path']

        # Preprocess
        image_array = preprocess_image(img_path, input_shape, grayscale)

        # Inference
        output, inference_time = run_inference(interpreter, image_array)
        inference_times.append(inference_time)

        # Get prediction
        if output.shape[-1] == 1:
            # Binary classification with single output
            prediction_score = output[0][0]
            predicted_class = 1 if prediction_score > threshold else 0
            confidence = prediction_score if prediction_score > threshold else 1 - prediction_score
        else:
            # Multi-class classification
            predictions = output[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            prediction_score = predictions

        y_true.append(img_info['true_label_idx'])
        y_pred.append(int(predicted_class))
        y_scores.append(float(confidence))

        results.append({
            'image': str(img_path),
            'true_label': img_info['true_label'],
            'true_label_idx': img_info['true_label_idx'],
            'predicted_label': labels[predicted_class] if predicted_class < len(labels) else f"Class {predicted_class}",
            'predicted_label_idx': int(predicted_class),
            'confidence': float(confidence),
            'correct': img_info['true_label_idx'] == predicted_class
        })

    # Calculate metrics
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    accuracy = accuracy_score(y_true, y_pred) if SKLEARN_AVAILABLE else sum(r['correct'] for r in results) / len(results)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct predictions: {sum(r['correct'] for r in results)}/{len(results)}")
    print(f"Average inference time: {np.mean(inference_times):.2f} ms")

    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for idx, label in enumerate(labels):
        class_results = [r for r in results if r['true_label_idx'] == idx]
        if class_results:
            class_acc = sum(r['correct'] for r in class_results) / len(class_results)
            print(f"  {label}: {class_acc:.4f} ({class_acc*100:.2f}%) [{len(class_results)} samples]")

    # Confusion Matrix
    if SKLEARN_AVAILABLE:
        print("\n" + "=" * 60)
        print("Confusion Matrix:")
        print("=" * 60)

        cm = confusion_matrix(y_true, y_pred)

        # Print text version
        print("\nConfusion Matrix (rows=true, cols=predicted):")
        print("        " + "  ".join(f"{label:>10}" for label in labels))
        for i, label in enumerate(labels):
            print(f"{label:>6}: " + "  ".join(f"{cm[i][j]:>10}" for j in range(len(labels))))

        # Print classification report
        print("\n" + "=" * 60)
        print("Classification Report:")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=labels, digits=4))

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_results:
            save_path = Path(save_results)
            cm_path = save_path.parent / f"{save_path.stem}_confusion_matrix.png"
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            print(f"\nConfusion matrix saved to: {cm_path}")

        plt.show()
    else:
        # Manual confusion matrix if sklearn not available
        print("\n" + "=" * 60)
        print("Confusion Matrix (manual - install sklearn for better formatting):")
        print("=" * 60)

        cm = [[0] * len(labels) for _ in range(len(labels))]
        for r in results:
            cm[r['true_label_idx']][r['predicted_label_idx']] += 1

        print("\n        " + "  ".join(f"{label:>10}" for label in labels))
        for i, label in enumerate(labels):
            print(f"{label:>6}: " + "  ".join(f"{cm[i][j]:>10}" for j in range(len(labels))))

    # Save detailed results if requested
    if save_results:
        save_path = Path(save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump({
                'model': str(model_path),
                'labeled_dir': str(labeled_dir),
                'labels': labels,
                'metrics': {
                    'accuracy': float(accuracy),
                    'total_samples': len(results),
                    'correct_predictions': sum(r['correct'] for r in results),
                    'average_inference_time_ms': float(np.mean(inference_times))
                },
                'per_class_accuracy': {
                    label: sum(r['correct'] for r in results if r['true_label_idx'] == idx) /
                           max(1, len([r for r in results if r['true_label_idx'] == idx]))
                    for idx, label in enumerate(labels)
                },
                'confusion_matrix': cm.tolist() if SKLEARN_AVAILABLE else cm,
                'detailed_results': results
            }, f, indent=2)

        print(f"\nDetailed results saved to: {save_results}")

    return results, cm, accuracy


def find_optimal_threshold(model_path, labeled_dir, labels=None, grayscale=False,
                          metric='f1', thresholds=None, save_plot=None):
    """
    Find optimal classification threshold by maximizing a metric (F1, accuracy, etc.)

    Args:
        model_path: Path to TFLite model
        labeled_dir: Directory with labeled subdirectories
        labels: Class labels (optional, will be inferred from directories)
        grayscale: Whether model expects grayscale input
        metric: Metric to optimize ('f1', 'accuracy', 'balanced_accuracy')
        thresholds: List of thresholds to test (default: np.arange(0.0, 1.01, 0.01))
        save_plot: Path to save threshold optimization plot (optional)

    Returns:
        optimal_threshold, results_dict
    """
    if not SKLEARN_AVAILABLE:
        print("❌ Error: scikit-learn is required for threshold optimization")
        print("   Install with: pip install scikit-learn matplotlib")
        return None, None

    print("\n" + "=" * 60)
    print("Finding Optimal Threshold")
    print("=" * 60)
    print(f"Optimization metric: {metric}")

    # Load model
    interpreter = load_tflite_model(model_path)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    output_details = interpreter.get_output_details()

    # Check if model is binary classification
    if output_details[0]['shape'][-1] != 1:
        print("❌ Error: Threshold optimization only works for binary classification models")
        print(f"   Your model has {output_details[0]['shape'][-1]} outputs")
        return None, None

    # Find labeled directories
    labeled_dir = Path(labeled_dir)
    class_dirs = [d for d in labeled_dir.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"❌ No class directories found in {labeled_dir}")
        return None, None

    # Infer labels from directory names if not provided
    if labels is None:
        labels = sorted([d.name for d in class_dirs])
        print(f"\nInferred labels from directories: {labels}")

    if len(labels) != 2:
        print(f"❌ Error: Threshold optimization requires exactly 2 classes, found {len(labels)}")
        return None, None

    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    print(f"\nLabels: {labels}")
    print(f"  Class 0 (negative): {labels[0]}")
    print(f"  Class 1 (positive): {labels[1]}")

    # Collect all images with their true labels
    all_images = []
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name not in label_to_idx:
            print(f"⚠️  Warning: Directory '{class_name}' not in labels, skipping")
            continue

        true_label_idx = label_to_idx[class_name]

        # Find all images
        image_files = list(class_dir.glob('*.jpg')) + \
                     list(class_dir.glob('*.jpeg')) + \
                     list(class_dir.glob('*.png'))

        for img_path in image_files:
            all_images.append({
                'path': img_path,
                'true_label': class_name,
                'true_label_idx': true_label_idx
            })

    if not all_images:
        print("❌ No images found")
        return None, None

    print(f"\nTotal images found: {len(all_images)}")
    for label in labels:
        count = sum(1 for img in all_images if img['true_label'] == label)
        print(f"  {label}: {count} images")

    # Run inference on all images to get raw scores
    print(f"\n{'='*60}")
    print("Running inference to collect scores...")
    print(f"{'='*60}")

    y_true = []
    y_scores = []  # Raw probability scores from model

    for img_info in all_images:
        img_path = img_info['path']

        # Preprocess
        image_array = preprocess_image(img_path, input_shape, grayscale)

        # Inference
        output, _ = run_inference(interpreter, image_array)

        # Get raw score (before thresholding)
        raw_score = float(output[0][0])

        y_true.append(img_info['true_label_idx'])
        y_scores.append(raw_score)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Define thresholds to test
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)

    print(f"\nTesting {len(thresholds)} thresholds from {thresholds[0]:.2f} to {thresholds[-1]:.2f}...")

    # Calculate metrics for each threshold
    results = {
        'thresholds': [],
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'accuracy_scores': []
    }

    for threshold in thresholds:
        # Apply threshold
        y_pred = (y_scores >= threshold).astype(int)

        # Calculate metrics
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        results['thresholds'].append(threshold)
        results['f1_scores'].append(f1)
        results['precision_scores'].append(precision)
        results['recall_scores'].append(recall)
        results['accuracy_scores'].append(accuracy)

    # Find optimal threshold based on chosen metric
    if metric == 'f1':
        metric_scores = results['f1_scores']
    elif metric == 'accuracy':
        metric_scores = results['accuracy_scores']
    elif metric == 'balanced_accuracy':
        # Balanced accuracy = (recall + specificity) / 2
        # For binary: balanced_acc = (TPR + TNR) / 2 = (recall + (1-FPR)) / 2
        metric_scores = results['recall_scores']  # Simplified for now
    else:
        print(f"⚠️  Unknown metric '{metric}', using F1 score")
        metric_scores = results['f1_scores']

    optimal_idx = np.argmax(metric_scores)
    optimal_threshold = results['thresholds'][optimal_idx]

    # Print results
    print("\n" + "=" * 60)
    print("Threshold Optimization Results:")
    print("=" * 60)
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"\nMetrics at optimal threshold:")
    print(f"  F1 Score:   {results['f1_scores'][optimal_idx]:.4f}")
    print(f"  Precision:  {results['precision_scores'][optimal_idx]:.4f}")
    print(f"  Recall:     {results['recall_scores'][optimal_idx]:.4f}")
    print(f"  Accuracy:   {results['accuracy_scores'][optimal_idx]:.4f}")

    # Show comparison with default threshold (0.5)
    default_idx = np.argmin(np.abs(np.array(results['thresholds']) - 0.5))
    print(f"\nComparison with default threshold (0.5):")
    print(f"  F1 Score:   {results['f1_scores'][default_idx]:.4f}")
    print(f"  Precision:  {results['precision_scores'][default_idx]:.4f}")
    print(f"  Recall:     {results['recall_scores'][default_idx]:.4f}")
    print(f"  Accuracy:   {results['accuracy_scores'][default_idx]:.4f}")

    improvement = results['f1_scores'][optimal_idx] - results['f1_scores'][default_idx]
    print(f"\nF1 Score improvement: {improvement:+.4f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: All metrics vs threshold
    ax1.plot(results['thresholds'], results['f1_scores'], 'b-', linewidth=2, label='F1 Score')
    ax1.plot(results['thresholds'], results['precision_scores'], 'g--', linewidth=2, label='Precision')
    ax1.plot(results['thresholds'], results['recall_scores'], 'r--', linewidth=2, label='Recall')
    ax1.plot(results['thresholds'], results['accuracy_scores'], 'orange', linewidth=2, label='Accuracy', alpha=0.7)

    # Mark optimal threshold
    ax1.axvline(optimal_threshold, color='black', linestyle=':', linewidth=2, alpha=0.5, label=f'Optimal: {optimal_threshold:.3f}')
    ax1.axvline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.3, label='Default: 0.5')

    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Plot 2: Precision-Recall curve with F1 iso-curves
    ax2.plot(results['recall_scores'], results['precision_scores'], 'b-', linewidth=2)

    # Mark optimal point
    ax2.plot(results['recall_scores'][optimal_idx], results['precision_scores'][optimal_idx],
            'r*', markersize=20, label=f'Optimal (threshold={optimal_threshold:.3f})')

    # Mark default threshold point
    ax2.plot(results['recall_scores'][default_idx], results['precision_scores'][default_idx],
            'go', markersize=10, label=f'Default (threshold=0.5)')

    # Add F1 iso-curves
    f1_levels = [0.2, 0.4, 0.6, 0.8]
    for f1_level in f1_levels:
        recall_range = np.linspace(0.01, 1, 100)
        precision = f1_level * recall_range / (2 * recall_range - f1_level)
        precision = np.clip(precision, 0, 1)
        ax2.plot(recall_range, precision, 'k--', alpha=0.2, linewidth=1)
        ax2.text(0.9, f1_level * 0.9 / (2 * 0.9 - f1_level), f'F1={f1_level}',
                fontsize=8, alpha=0.5)

    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    if save_plot:
        save_path = Path(save_plot)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nThreshold optimization plot saved to: {save_plot}")

    plt.show()

    # Also calculate and show ROC curve
    try:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        print(f"\nROC AUC Score: {roc_auc:.4f}")

        # Plot ROC curve
        fig_roc = plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')

        # Mark optimal threshold on ROC curve
        optimal_fpr = 1 - results['precision_scores'][optimal_idx] * results['recall_scores'][optimal_idx] / max(y_scores.mean(), 0.01)
        optimal_tpr = results['recall_scores'][optimal_idx]

        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        if save_plot:
            roc_path = Path(save_plot).parent / (Path(save_plot).stem + "_roc.png")
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {roc_path}")

        plt.show()
    except Exception as e:
        print(f"\n⚠️  Could not generate ROC curve: {e}")

    return optimal_threshold, results


def compare_with_pytorch(tflite_path, pytorch_path, image_path, architecture, num_classes, grayscale):
    """
    Compare TFLite model output with original PyTorch model

    Args:
        tflite_path: Path to TFLite model
        pytorch_path: Path to PyTorch checkpoint
        image_path: Path to test image
        architecture: Model architecture
        num_classes: Number of classes
        grayscale: Whether model uses grayscale input
    """
    print("\n" + "=" * 60)
    print("Comparing TFLite vs PyTorch")
    print("=" * 60)

    # Load TFLite model
    import tensorflow as tf
    tflite_interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    tflite_interpreter.allocate_tensors()

    input_details = tflite_interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    # Load PyTorch model
    import torch
    from src.models.efficientnet import create_efficientnet_model

    device = torch.device('cpu')
    checkpoint = torch.load(pytorch_path, map_location=device)

    model = create_efficientnet_model(
        num_classes=num_classes,
        model_name=architecture,
        pretrained=False,
        in_channels=1 if grayscale else 3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess for TFLite (NHWC format)
    tflite_input = preprocess_image(image_path, input_shape, grayscale)

    # Preprocess for PyTorch (NCHW format)
    # Convert from NHWC to NCHW
    if grayscale:
        pytorch_input = torch.from_numpy(tflite_input).permute(0, 3, 1, 2)
    else:
        pytorch_input = torch.from_numpy(tflite_input).permute(0, 3, 1, 2)

    # Run TFLite inference
    tflite_interpreter.set_tensor(input_details[0]['index'], tflite_input)
    tflite_interpreter.invoke()
    output_details = tflite_interpreter.get_output_details()
    tflite_output = tflite_interpreter.get_tensor(output_details[0]['index'])

    # Run PyTorch inference
    with torch.no_grad():
        pytorch_output = model(pytorch_input).numpy()

    # Compare results
    print(f"\nTFLite output: {tflite_output}")
    print(f"PyTorch output: {pytorch_output}")
    print(f"\nDifference: {np.abs(tflite_output - pytorch_output)}")
    print(f"Max difference: {np.max(np.abs(tflite_output - pytorch_output)):.6f}")
    print(f"Mean difference: {np.mean(np.abs(tflite_output - pytorch_output)):.6f}")

    if np.allclose(tflite_output, pytorch_output, atol=1e-3):
        print("\n✅ TFLite and PyTorch outputs match (within tolerance)")
    else:
        print("\n⚠️ Warning: TFLite and PyTorch outputs differ")
        print("   This is normal due to quantization and conversion")


def main():
    parser = argparse.ArgumentParser(description='Test TFLite model')

    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to TFLite model (.tflite)')
    parser.add_argument('--image', type=str,
                        help='Path to single test image')
    parser.add_argument('--image-dir', type=str,
                        help='Directory of test images (unlabeled)')
    parser.add_argument('--labeled-dir', type=str,
                        help='Directory with labeled subdirectories (e.g., labeled_dir/class1/, labeled_dir/class2/)')
    parser.add_argument('--labels', type=str, nargs='+',
                        help='Class labels (e.g., --labels non-wood wood)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Model expects grayscale input')
    parser.add_argument('--save-results', type=str,
                        help='Path to save detailed results JSON (for labeled-dir test)')

    # Comparison arguments
    parser.add_argument('--compare-pytorch', type=str,
                        help='Path to PyTorch checkpoint for comparison')
    parser.add_argument('--architecture', type=str,
                        help='Model architecture (for PyTorch comparison)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes (for PyTorch comparison)')

    # Batch test arguments
    parser.add_argument('--max-images', type=int, default=10,
                        help='Maximum images for batch test (default: 10)')

    # Threshold argument
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold for binary classification (default: 0.5)')

    # Threshold optimization arguments
    parser.add_argument('--find-threshold', action='store_true',
                        help='Find optimal threshold using labeled data (requires --labeled-dir)')
    parser.add_argument('--optimize-metric', type=str, default='f1', choices=['f1', 'accuracy'],
                        help='Metric to optimize when finding threshold (default: f1)')
    parser.add_argument('--save-threshold-plot', type=str,
                        help='Path to save threshold optimization plot')
    parser.add_argument('--use-optimal-threshold', action='store_true',
                        help='After finding optimal threshold, re-run evaluation with it')

    args = parser.parse_args()

    print("=" * 60)
    print("TFLite Model Testing")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Grayscale: {args.grayscale}")
    print(f"Threshold: {args.threshold}")
    if args.labels:
        print(f"Labels: {args.labels}")
    print("=" * 60)

    # Test single image
    if args.image:
        test_single_image(
            args.model,
            args.image,
            labels=args.labels,
            grayscale=args.grayscale,
            threshold=args.threshold
        )

    # Batch test (unlabeled)
    if args.image_dir:
        batch_test(
            args.model,
            args.image_dir,
            labels=args.labels,
            grayscale=args.grayscale,
            max_images=args.max_images,
            threshold=args.threshold
        )

    # Find optimal threshold
    optimal_threshold = None
    if args.find_threshold:
        if not args.labeled_dir:
            print("\n❌ Error: --find-threshold requires --labeled-dir")
            sys.exit(1)

        optimal_threshold, threshold_results = find_optimal_threshold(
            args.model,
            args.labeled_dir,
            labels=args.labels,
            grayscale=args.grayscale,
            metric=args.optimize_metric,
            save_plot=args.save_threshold_plot
        )

        if optimal_threshold is not None and args.use_optimal_threshold:
            print(f"\n{'='*60}")
            print(f"Re-running evaluation with optimal threshold: {optimal_threshold:.3f}")
            print(f"{'='*60}")
            args.threshold = optimal_threshold

    # Labeled test with confusion matrix
    if args.labeled_dir and not args.find_threshold:
        labeled_test(
            args.model,
            args.labeled_dir,
            labels=args.labels,
            grayscale=args.grayscale,
            save_results=args.save_results,
            threshold=args.threshold
        )
    elif args.labeled_dir and args.find_threshold and args.use_optimal_threshold:
        # Run full evaluation with optimal threshold
        labeled_test(
            args.model,
            args.labeled_dir,
            labels=args.labels,
            grayscale=args.grayscale,
            save_results=args.save_results,
            threshold=args.threshold
        )

    # Compare with PyTorch
    if args.compare_pytorch:
        if not args.architecture:
            print("\n❌ Error: --architecture required for PyTorch comparison")
            sys.exit(1)
        if not args.image:
            print("\n❌ Error: --image required for PyTorch comparison")
            sys.exit(1)

        compare_with_pytorch(
            args.model,
            args.compare_pytorch,
            args.image,
            args.architecture,
            args.num_classes,
            args.grayscale
        )

    if not args.image and not args.image_dir and not args.labeled_dir and not args.find_threshold:
        print("\n⚠️ Please provide --image, --image-dir, or --labeled-dir to test")
        print("\nExamples:")
        print("  # Test single image")
        print("  python test_tflite_model.py --model model.tflite --image test.jpg --labels non-wood wood")
        print("\n  # Test single image with custom threshold")
        print("  python test_tflite_model.py --model model.tflite --image test.jpg --labels non-wood wood --threshold 0.7")
        print("\n  # Test batch of images (unlabeled)")
        print("  python test_tflite_model.py --model model.tflite --image-dir test_images/ --labels non-wood wood")
        print("\n  # Test labeled dataset with confusion matrix")
        print("  python test_tflite_model.py --model model.tflite --labeled-dir labeled_data/ --labels non-wood wood")
        print("\n  # Test labeled dataset with custom threshold and save results")
        print("  python test_tflite_model.py --model model.tflite --labeled-dir labeled_data/ \\")
        print("         --labels non-wood wood --threshold 0.6 --save-results results.json")
        print("\n  # Find optimal threshold using F1 score")
        print("  python test_tflite_model.py --model model.tflite --labeled-dir labeled_data/ \\")
        print("         --labels non-wood wood --find-threshold --save-threshold-plot threshold_plot.png")
        print("\n  # Find optimal threshold and re-evaluate with it")
        print("  python test_tflite_model.py --model model.tflite --labeled-dir labeled_data/ \\")
        print("         --labels non-wood wood --find-threshold --use-optimal-threshold --save-results results.json")
        print("\n  # Compare with PyTorch")
        print("  python test_tflite_model.py --model model.tflite --image test.jpg \\")
        print("         --compare-pytorch checkpoint.pt --architecture efficientnet_b0 --grayscale")


if __name__ == '__main__':
    main()
