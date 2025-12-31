"""
Convert PyTorch model (.pt) to TFLite format for React Native deployment

This script converts PyTorch models to TFLite format compatible with:
- react-native-fast-tflite (v1.6.1)
- @tensorflow/tfjs (v4.1.0)

Conversion path: PyTorch → ONNX → TensorFlow → TFLite

References:
- https://github.com/mrousavy/react-native-fast-tflite
- https://github.com/PINTO0309/onnx2tf
- https://medium.com/@amitvermaphd/converting-pytorch-models-to-onnx-and-tflite-for-mobile-apps-bf903d54ba0e
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import json


def export_to_onnx(model, dummy_input, output_path, input_names=None, output_names=None, opset_version=12):
    """
    Export PyTorch model to ONNX format

    Args:
        model: PyTorch model
        dummy_input: Example input tensor (or tuple of tensors)
        output_path: Output ONNX file path
        input_names: List of input names (default: ['input'])
        output_names: List of output names (default: ['output'])
        opset_version: ONNX opset version (12 recommended for compatibility)

    Returns:
        Path to ONNX file
    """
    print("\n" + "=" * 60)
    print("Step 1: Exporting PyTorch to ONNX")
    print("=" * 60)

    model.eval()

    # Default names
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        }
    )

    print(f"✅ ONNX model saved to: {output_path}")
    print(f"   Opset version: {opset_version}")

    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model verification passed")
    except ImportError:
        print("⚠ onnx not installed, skipping verification")
    except Exception as e:
        print(f"⚠ ONNX verification warning: {e}")

    return output_path


def convert_onnx_to_tensorflow(onnx_path, output_dir):
    """
    Convert ONNX model to TensorFlow SavedModel

    Uses onnx2tf (recommended for 2025) or falls back to onnx-tensorflow

    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory for TensorFlow SavedModel

    Returns:
        Path to SavedModel directory
    """
    print("\n" + "=" * 60)
    print("Step 2: Converting ONNX to TensorFlow SavedModel")
    print("=" * 60)

    # Import onnx2tf (let errors show naturally)
    import onnx2tf
    print("Using onnx2tf")

    # Convert using onnx2tf
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        output_signaturedefs=True,
        copy_onnx_input_output_names_to_tflite=True,
    )

    # Find the saved model directory (onnx2tf may create it with different names)
    possible_paths = [
        output_dir / "saved_model",
        output_dir,  # Sometimes saves directly in output_dir
        output_dir / "model",
    ]

    saved_model_path = None
    for path in possible_paths:
        # Check if saved_model.pb or saved_model.pbtxt exists
        if (path / "saved_model.pb").exists() or (path / "saved_model.pbtxt").exists():
            saved_model_path = path
            break
        # Also check subdirectories
        for subdir in path.iterdir():
            if subdir.is_dir():
                if (subdir / "saved_model.pb").exists() or (subdir / "saved_model.pbtxt").exists():
                    saved_model_path = subdir
                    break
        if saved_model_path:
            break

    if not saved_model_path:
        print(f"\n❌ Error: Could not find SavedModel in {output_dir}")
        print(f"Contents of {output_dir}:")
        for item in output_dir.rglob("*"):
            print(f"  {item}")
        sys.exit(1)

    print(f"✅ TensorFlow SavedModel found at: {saved_model_path}")

    return saved_model_path


def convert_savedmodel_to_tflite(saved_model_path, output_path, quantize='dynamic', input_shape=None):
    """
    Convert TensorFlow SavedModel to TFLite

    Args:
        saved_model_path: Path to SavedModel directory
        output_path: Output TFLite file path
        quantize: Quantization type ('none', 'dynamic', 'float16', 'int8')
        input_shape: Input shape for optimization (optional)

    Returns:
        Path to TFLite file
    """
    print("\n" + "=" * 60)
    print("Step 3: Converting TensorFlow to TFLite")
    print("=" * 60)

    import tensorflow as tf

    # Create converter
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))

    # Quantization settings
    if quantize == 'dynamic':
        print("Applying dynamic range quantization (reduces model size)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif quantize == 'float16':
        print("Applying float16 quantization (reduces size, maintains accuracy)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quantize == 'int8':
        print("Applying int8 quantization (smallest size, may reduce accuracy)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_dataset():
            # Generate dummy data for calibration
            if input_shape:
                for _ in range(100):
                    yield [np.random.rand(*input_shape).astype(np.float32)]

        if input_shape:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

    elif quantize == 'none':
        print("No quantization applied (largest size, best accuracy)...")

    # Additional optimizations for mobile deployment
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True

    # Convert
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Try a different opset_version (11 or 13)")
        print("  2. Simplify your model (remove unsupported operations)")
        print("  3. Check TensorFlow/ONNX compatibility")
        raise

    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # Get model size
    size_mb = len(tflite_model) / (1024 * 1024)

    print(f"✅ TFLite model saved to: {output_path}")
    print(f"   Model size: {size_mb:.2f} MB")

    if size_mb > 50:
        print("⚠ Warning: Model size > 50MB may cause issues on budget devices")
        print("   Consider using more aggressive quantization (float16 or int8)")
    elif size_mb > 100:
        print("⚠ Warning: Model size > 100MB requires careful optimization")

    return output_path


def validate_tflite_model(tflite_path, dummy_input_np):
    """
    Validate TFLite model by running inference

    Args:
        tflite_path: Path to TFLite model
        dummy_input_np: Example input as numpy array (NHWC format)

    Returns:
        Model metadata (input/output shapes, types)
    """
    print("\n" + "=" * 60)
    print("Step 4: Validating TFLite Model")
    print("=" * 60)

    import tensorflow as tf

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nModel Input Details:")
    for i, input_detail in enumerate(input_details):
        print(f"  Input {i}:")
        print(f"    Name: {input_detail['name']}")
        print(f"    Shape: {input_detail['shape']}")
        print(f"    Type: {input_detail['dtype']}")

    print("\nModel Output Details:")
    for i, output_detail in enumerate(output_details):
        print(f"  Output {i}:")
        print(f"    Name: {output_detail['name']}")
        print(f"    Shape: {output_detail['shape']}")
        print(f"    Type: {output_detail['dtype']}")

    # Run test inference
    print("\nRunning test inference...")

    # Ensure correct shape and type
    test_input = dummy_input_np.astype(input_details[0]['dtype'])

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    print(f"✅ Inference successful!")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Return metadata
    metadata = {
        'input': {
            'name': input_details[0]['name'],
            'shape': input_details[0]['shape'].tolist(),
            'dtype': str(input_details[0]['dtype'])
        },
        'output': {
            'name': output_details[0]['name'],
            'shape': output_details[0]['shape'].tolist(),
            'dtype': str(output_details[0]['dtype'])
        }
    }

    return metadata


def create_model_info(metadata, output_path, labels=None, preprocessing_info=None):
    """
    Create model info JSON file for React Native integration

    Args:
        metadata: Model metadata from validation
        output_path: Output JSON file path
        labels: Class labels (optional)
        preprocessing_info: Preprocessing information (optional)
    """
    print("\n" + "=" * 60)
    print("Creating Model Info File")
    print("=" * 60)

    model_info = {
        'input': metadata['input'],
        'output': metadata['output'],
        'usage': {
            'react_native': 'react-native-fast-tflite v1.6.1+',
            'tfjs': '@tensorflow/tfjs v4.1.0+'
        }
    }

    if labels:
        model_info['labels'] = labels

    if preprocessing_info:
        model_info['preprocessing'] = preprocessing_info
    else:
        model_info['preprocessing'] = {
            'note': 'Update this based on your model training',
            'input_format': 'NHWC (batch, height, width, channels)',
            'example': 'Normalize to [0, 1] or [-1, 1] depending on training'
        }

    with open(output_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"✅ Model info saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch model to TFLite for React Native',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convert_to_tflite.py --model model.pt --input-shape 1 3 224 224

  # With quantization
  python convert_to_tflite.py --model model.pt --input-shape 1 3 224 224 --quantize float16

  # Custom output name
  python convert_to_tflite.py --model model.pt --input-shape 1 3 224 224 --output-name my_model

  # From checkpoint (state_dict)
  python convert_to_tflite.py --model checkpoint.pt --input-shape 1 1 224 224 \
      --architecture efficientnet_b0 --num-classes 2 --grayscale
        """
    )

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to PyTorch model (.pt or .pth file)')
    parser.add_argument('--input-shape', type=int, nargs='+', required=True,
                        help='Input shape (e.g., 1 3 224 224 for batch=1, channels=3, H=224, W=224)')

    # Model architecture (for loading from state_dict)
    parser.add_argument('--architecture', type=str,
                        choices=['resnet18', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                                'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                                'efficientnet_b6', 'efficientnet_b7'],
                        help='Model architecture (required if checkpoint only has state_dict)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of output classes (default: 2)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Model uses grayscale input (1 channel)')

    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='outputs/tflite',
                        help='Output directory (default: outputs/tflite)')
    parser.add_argument('--output-name', type=str, default='model',
                        help='Output model name (default: model)')
    parser.add_argument('--quantize', type=str, default='dynamic',
                        choices=['none', 'dynamic', 'float16', 'int8'],
                        help='Quantization type (default: dynamic)')
    parser.add_argument('--opset-version', type=int, default=12,
                        help='ONNX opset version (default: 12)')
    parser.add_argument('--labels', type=str, nargs='+',
                        help='Class labels (e.g., --labels cat dog bird)')

    args = parser.parse_args()

    # Validate input shape
    if any(dim < 1 for dim in args.input_shape):
        print("❌ Error: Input shape dimensions must be positive integers")
        print(f"   Got: {args.input_shape}")
        print("\n   Use 1 for batch size (not -1)")
        print("   Example: --input-shape 1 3 224 224")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PyTorch to TFLite Conversion")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input shape: {args.input_shape}")
    print(f"Quantization: {args.quantize}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Load PyTorch model
    print("\nLoading PyTorch model...")
    device = torch.device('cpu')  # Always use CPU for conversion

    checkpoint = torch.load(args.model, map_location=device)

    # Handle different model formats
    if isinstance(checkpoint, dict):
        # Checkpoint contains state_dict, need to instantiate model
        if 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
            print("📦 Checkpoint contains state_dict, instantiating model...")

            # Check if architecture is specified
            if not args.architecture:
                print("\n❌ Error: Checkpoint contains only state_dict")
                print("Please specify the model architecture:")
                print("  --architecture resnet18|efficientnet_b0-b7")
                print("  --num-classes N")
                print("  --grayscale (if model uses 1-channel input)")
                print("\nExample:")
                print("  python convert_to_tflite.py --model checkpoint.pt \\")
                print("      --input-shape 1 1 224 224 \\")
                print("      --architecture efficientnet_b0 \\")
                print("      --num-classes 2 \\")
                print("      --grayscale")
                sys.exit(1)

            # Import model classes
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))

            # Create model based on architecture
            print(f"   Architecture: {args.architecture}")
            print(f"   Input channels: {1 if args.grayscale else 3}")
            print(f"   Num classes: {args.num_classes}")

            if args.architecture == 'resnet18':
                from src.models.resnet import ResNet18
                model = ResNet18(
                    num_classes=args.num_classes,
                    pretrained=False,
                    in_channels=1 if args.grayscale else 3
                )
            elif args.architecture == 'resnet50':
                print("❌ Error: ResNet50 not implemented in your codebase")
                print("   Available: resnet18, efficientnet_b0-b7")
                sys.exit(1)
            elif 'efficientnet' in args.architecture:
                from src.models.efficientnet import create_efficientnet_model
                model = create_efficientnet_model(
                    num_classes=args.num_classes,
                    model_name=args.architecture,
                    pretrained=False,
                    in_channels=1 if args.grayscale else 3
                )
            else:
                print(f"❌ Unknown architecture: {args.architecture}")
                sys.exit(1)

            # Load state dict
            state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
            model.load_state_dict(checkpoint[state_dict_key])
            model.to(device)
            model.eval()
            print(f"✅ Model loaded from state_dict")

        else:
            print("❌ Unknown checkpoint format")
            print(f"   Keys found: {checkpoint.keys()}")
            sys.exit(1)
    else:
        # Full model saved directly
        model = checkpoint
        model.eval()
        print(f"✅ Model loaded successfully")

    # Create dummy input
    dummy_input = torch.randn(*args.input_shape).to(device)
    print(f"   Dummy input shape: {dummy_input.shape}")

    # Step 1: Export to ONNX
    onnx_path = output_dir / f"{args.output_name}.onnx"
    export_to_onnx(model, dummy_input, onnx_path, opset_version=args.opset_version)

    # Step 2: Convert ONNX to TensorFlow
    tf_dir = output_dir / "tensorflow"
    tf_dir.mkdir(exist_ok=True)
    saved_model_path = convert_onnx_to_tensorflow(onnx_path, tf_dir)

    # Step 3: Convert to TFLite
    tflite_path = output_dir / f"{args.output_name}.tflite"

    # Prepare input shape for quantization (convert NCHW to NHWC)
    if len(args.input_shape) == 4:
        # PyTorch: (batch, channels, height, width)
        # TFLite: (batch, height, width, channels)
        nhwc_shape = [args.input_shape[0], args.input_shape[2], args.input_shape[3], args.input_shape[1]]
    else:
        nhwc_shape = args.input_shape

    convert_savedmodel_to_tflite(
        saved_model_path,
        tflite_path,
        quantize=args.quantize,
        input_shape=nhwc_shape if args.quantize == 'int8' else None
    )

    # Step 4: Validate
    # Convert dummy input to NHWC for TFLite
    if len(dummy_input.shape) == 4:
        dummy_input_np = dummy_input.permute(0, 2, 3, 1).cpu().numpy()
    else:
        dummy_input_np = dummy_input.cpu().numpy()

    metadata = validate_tflite_model(tflite_path, dummy_input_np)

    # Step 5: Create model info JSON
    info_path = output_dir / f"{args.output_name}_info.json"
    create_model_info(metadata, info_path, labels=args.labels)

    # Final summary
    print("\n" + "=" * 60)
    print("✅ Conversion Complete!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  1. TFLite model:    {tflite_path}")
    print(f"  2. Model info:      {info_path}")
    print(f"  3. ONNX model:      {onnx_path} (intermediate)")
    print(f"  4. SavedModel:      {saved_model_path} (intermediate)")

    print("\n📱 React Native Integration Guide:")
    print("=" * 60)
    print(f"\n1. Install dependencies:")
    print(f"   npm install react-native-fast-tflite@1.6.1")

    print(f"\n2. Copy model to your React Native project:")
    print(f"   cp {tflite_path} <your-rn-project>/assets/")

    print(f"\n3. Load and use the model:")
    print(f"""
   import {{ useTensorflowModel }} from 'react-native-fast-tflite';

   // Load model
   const model = useTensorflowModel(
     require('./assets/{tflite_path.name}')
   );

   // Prepare input (shape: {metadata['input']['shape']})
   const inputData = new Float32Array({np.prod(metadata['input']['shape'])});
   // ... fill inputData with your preprocessed image data ...

   // Run inference
   const output = model.run(inputData);
   // Output shape: {metadata['output']['shape']}

   console.log('Predictions:', output);
""")

    print("\n⚠ Important Notes:")
    print("=" * 60)
    print("  1. Input format: TFLite uses NHWC (batch, height, width, channels)")
    print("  2. PyTorch uses NCHW - you must transpose in your app!")
    print("  3. Preprocessing: Apply the same normalization as during training")
    print("  4. Quantized models: Input/output may be uint8 instead of float32")
    print(f"  5. Your model expects: {metadata['input']['dtype']}")

    print("\n🔧 Troubleshooting:")
    print("=" * 60)
    print("  If conversion fails:")
    print("  - Try different opset_version (11, 12, or 13)")
    print("  - Check for unsupported PyTorch operations")
    print("  - Simplify your model architecture")
    print("  - Install latest: pip install onnx2tf tensorflow onnx")


if __name__ == '__main__':
    main()
