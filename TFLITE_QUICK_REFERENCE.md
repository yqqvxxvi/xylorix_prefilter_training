# TFLite Conversion Quick Reference

## One-Command Conversion

```bash
python scripts/convert_to_tflite.py \
    --model your_model.pt \
    --input-shape 1 3 224 224 \
    --quantize float16 \
    --output-name my_model \
    --labels class1 class2 class3
```

## Installation

```bash
pip install -r requirements-tflite.txt
```

## Common Use Cases

### RGB Image Model (224x224)
```bash
python scripts/convert_to_tflite.py \
    --model model.pt \
    --input-shape 1 3 224 224
```

### Grayscale Image Model (224x224)
```bash
python scripts/convert_to_tflite.py \
    --model model.pt \
    --input-shape 1 1 224 224
```

### With Compression (Recommended)
```bash
python scripts/convert_to_tflite.py \
    --model model.pt \
    --input-shape 1 3 224 224 \
    --quantize float16  # 50% size reduction
```

### Maximum Compression
```bash
python scripts/convert_to_tflite.py \
    --model model.pt \
    --input-shape 1 3 224 224 \
    --quantize int8  # 75% size reduction
```

## React Native Usage

### Load Model
```javascript
import { useTensorflowModel } from 'react-native-fast-tflite';

const model = useTensorflowModel(
  require('./assets/my_model.tflite')
);
```

### Run Inference
```javascript
// Prepare input (224x224x3 image, normalized to [0, 1])
const input = new Float32Array(224 * 224 * 3);
// ... fill with image data ...

// Predict
const output = model.run(input);

// Get class
const predictions = Array.from(output);
const classIndex = predictions.indexOf(Math.max(...predictions));
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Unsupported operation" | Try `--opset-version 11` or `--opset-version 13` |
| "onnx2tf not found" | Run `pip install onnx2tf` |
| Model too large | Use `--quantize float16` or `--quantize int8` |
| Low accuracy after conversion | Use `--quantize dynamic` instead of `int8` |
| TensorFlow version error | Use Python 3.8-3.10, not 3.11+ |

## Format Differences

| Framework | Format | Example |
|-----------|--------|---------|
| PyTorch | NCHW (channels first) | (1, 3, 224, 224) |
| TFLite | NHWC (channels last) | (1, 224, 224, 3) |

**Note**: Conversion script handles this automatically!

## Model Size Guidelines

- **< 5MB**: ✅ Excellent
- **5-25MB**: ✅ Good  
- **25-50MB**: ⚠️ Test on budget devices
- **50-100MB**: ⚠️ May have issues
- **> 100MB**: ❌ Needs optimization

## Quantization Comparison

| Type | Size | Speed | Accuracy |
|------|------|-------|----------|
| none | 100% | 1x | Best |
| dynamic | ~75% | 1.5x | Good |
| float16 | ~50% | 2x | Good |
| int8 | ~25% | 3x | Fair |

## Web Resources

- [react-native-fast-tflite](https://github.com/mrousavy/react-native-fast-tflite)
- [onnx2tf](https://github.com/PINTO0309/onnx2tf)
- [TFLite Guide](https://www.tensorflow.org/lite)
- [Conversion Tutorial](https://medium.com/@amitvermaphd/converting-pytorch-models-to-onnx-and-tflite-for-mobile-apps-bf903d54ba0e)
