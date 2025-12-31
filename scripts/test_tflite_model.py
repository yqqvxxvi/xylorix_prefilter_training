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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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


def run_inference(interpreter, image_array):
    """
    Run inference on preprocessed image

    Args:
        interpreter: TFLite interpreter
        image_array: Preprocessed image array

    Returns:
        Output predictions
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

    return output, inference_time


def test_single_image(model_path, image_path, labels=None, grayscale=False):
    """
    Test model on a single image

    Args:
        model_path: Path to TFLite model
        image_path: Path to test image
        labels: Class labels (optional)
        grayscale: Whether model expects grayscale input
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
    print(f"Raw output: {output}")

    # Interpret output
    if output.shape[-1] == 1:
        # Single output (binary classification, 0-1)
        prediction = output[0][0]
        print(f"\nPrediction value: {prediction:.4f}")
        if labels and len(labels) == 2:
            if prediction > 0.5:
                print(f"Predicted class: {labels[1]} (confidence: {prediction:.2%})")
            else:
                print(f"Predicted class: {labels[0]} (confidence: {1-prediction:.2%})")
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


def batch_test(model_path, image_dir, labels=None, grayscale=False, max_images=10):
    """
    Test model on multiple images

    Args:
        model_path: Path to TFLite model
        image_dir: Directory containing test images
        labels: Class labels (optional)
        grayscale: Whether model expects grayscale input
        max_images: Maximum number of images to test
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
            predicted_class = 1 if prediction > 0.5 else 0
            confidence = prediction if prediction > 0.5 else 1 - prediction
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
                        help='Directory of test images')
    parser.add_argument('--labels', type=str, nargs='+',
                        help='Class labels (e.g., --labels non-wood wood)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Model expects grayscale input')

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

    args = parser.parse_args()

    print("=" * 60)
    print("TFLite Model Testing")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Grayscale: {args.grayscale}")
    if args.labels:
        print(f"Labels: {args.labels}")
    print("=" * 60)

    # Test single image
    if args.image:
        test_single_image(
            args.model,
            args.image,
            labels=args.labels,
            grayscale=args.grayscale
        )

    # Batch test
    if args.image_dir:
        batch_test(
            args.model,
            args.image_dir,
            labels=args.labels,
            grayscale=args.grayscale,
            max_images=args.max_images
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

    if not args.image and not args.image_dir:
        print("\n⚠️ Please provide --image or --image-dir to test")
        print("\nExamples:")
        print("  # Test single image")
        print("  python test_tflite_model.py --model model.tflite --image test.jpg --labels non-wood wood")
        print("\n  # Test batch of images")
        print("  python test_tflite_model.py --model model.tflite --image-dir test_images/ --labels non-wood wood")
        print("\n  # Compare with PyTorch")
        print("  python test_tflite_model.py --model model.tflite --image test.jpg \\")
        print("         --compare-pytorch checkpoint.pt --architecture efficientnet_b0 --grayscale")


if __name__ == '__main__':
    main()
