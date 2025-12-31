# PyTorch to TFLite Conversion Guide

Complete guide for converting PyTorch models to TFLite format for React Native deployment.

## Prerequisites

### Install Required Packages

```bash
# Core dependencies
pip install torch torchvision
pip install onnx
pip install tensorflow

# Conversion tools (choose one)
pip install onnx2tf  # Recommended for 2025
# OR
pip install onnx-tf  # Alternative (older)

# Optional but recommended
pip install onnx-simplifier  # Simplifies ONNX models
```

### Version Compatibility

Based on research, these versions work well together:

```bash
# Python 3.8-3.10 (avoid 3.11+ due to TensorFlow compatibility)
pip install torch>=1.13.0
pip install onnx>=1.14.0
pip install tensorflow>=2.12.0
pip install onnx2tf>=1.16.0
```

## Quick Start

### 1. Save Your PyTorch Model

Make sure you save the **entire model**, not just the state_dict:

```python
# ✅ CORRECT: Save entire model
torch.save(model, 'model.pt')

# ❌ WRONG: Only saves weights
torch.save(model.state_dict(), 'model.pt')
```

### 2. Run Conversion

```bash
python scripts/convert_to_tflite.py \
    --model path/to/your/model.pt \
    --input-shape 1 3 224 224 \
    --quantize float16 \
    --output-name my_model
```

**Parameters explained:**
- `--model`: Path to your PyTorch model (.pt file)
- `--input-shape`: Input tensor shape (batch, channels, height, width)
  - Example: `1 3 224 224` = batch=1, RGB (3 channels), 224x224 image
  - Example: `1 1 224 224` = batch=1, grayscale (1 channel), 224x224
- `--quantize`: Compression method
  - `dynamic`: Good balance (recommended)
  - `float16`: Better compression, maintains accuracy
  - `int8`: Smallest size, may reduce accuracy
  - `none`: No compression, largest size
- `--output-name`: Name for output files

### 3. Get Your TFLite Model

The script creates:
- `my_model.tflite` - Your converted model
- `my_model_info.json` - Model metadata (shapes, types)
- `my_model.onnx` - Intermediate ONNX model
- `tensorflow/saved_model/` - Intermediate TF model

## React Native Integration

### Option 1: react-native-fast-tflite (Recommended)

#### Installation

```bash
npm install react-native-fast-tflite@1.6.1
```

#### Basic Usage

```javascript
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useResizePlugin } from 'react-native-fast-tflite';

function App() {
  // Load model
  const model = useTensorflowModel(
    require('./assets/my_model.tflite')
  );

  const resize = useResizePlugin();

  async function classifyImage(imagePath) {
    // 1. Resize image to model input size (224x224)
    const resized = await resize(imagePath, {
      width: 224,
      height: 224,
      pixelFormat: 'rgb'
    });

    // 2. Preprocess (normalize to [0, 1])
    const inputData = new Float32Array(resized.length);
    for (let i = 0; i < resized.length; i++) {
      inputData[i] = resized[i] / 255.0;
    }

    // 3. Run inference
    const output = model.run(inputData);

    // 4. Get prediction
    const predictions = Array.from(output);
    const maxIndex = predictions.indexOf(Math.max(...predictions));

    return {
      class: maxIndex,
      confidence: predictions[maxIndex],
      probabilities: predictions
    };
  }

  // Use in component
  return (
    <Button
      title="Classify"
      onPress={async () => {
        const result = await classifyImage('path/to/image.jpg');
        console.log('Prediction:', result);
      }}
    />
  );
}
```

### Option 2: TensorFlow.js

#### Installation

```bash
npm install @tensorflow/tfjs@4.1.0
npm install @tensorflow/tfjs-react-native
```

#### Usage

```javascript
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';

async function loadAndPredict() {
  // Wait for TF to be ready
  await tf.ready();

  // Load model
  const model = await tf.loadGraphModel(
    'https://your-server.com/model.tflite'
  );

  // Prepare input (224x224 RGB image)
  const imageTensor = tf.browser.fromPixels(imageElement)
    .resizeBilinear([224, 224])
    .expandDims(0)
    .div(255.0);

  // Run inference
  const predictions = await model.predict(imageTensor);

  // Get results
  const probabilities = await predictions.data();
  console.log('Predictions:', probabilities);
}
```

## Model Input/Output Format

### Input Format Differences

**PyTorch uses NCHW** (channels first):
```
Shape: (batch, channels, height, width)
Example: (1, 3, 224, 224)
```

**TFLite uses NHWC** (channels last):
```
Shape: (batch, height, width, channels)
Example: (1, 224, 224, 3)
```

### Conversion Handles This!

The conversion script **automatically transposes** from NCHW to NHWC, so your TFLite model expects NHWC format.

### In Your React Native App

```javascript
// Your image data is typically already in NHWC format
// (height, width, channels) from camera or file

// Example: 224x224 RGB image
const imageData = new Uint8Array(224 * 224 * 3);
// Fill imageData with pixel values...

// Normalize to [0, 1]
const inputData = new Float32Array(imageData.length);
for (let i = 0; i < imageData.length; i++) {
  inputData[i] = imageData[i] / 255.0;
}

// Ready for model!
const output = model.run(inputData);
```

## Preprocessing

### Important: Match Training Preprocessing!

Your mobile app must use the **same preprocessing** as training:

```javascript
// Example: ImageNet normalization
const mean = [0.485, 0.456, 0.406];
const std = [0.229, 0.224, 0.225];

function preprocessImage(imageData, width, height) {
  const input = new Float32Array(width * height * 3);

  for (let i = 0; i < width * height; i++) {
    // RGB channels
    for (let c = 0; c < 3; c++) {
      const idx = i * 3 + c;
      // Normalize: (pixel / 255 - mean) / std
      input[idx] = (imageData[idx] / 255.0 - mean[c]) / std[c];
    }
  }

  return input;
}
```

## Troubleshooting

### Error: "Unsupported operation"

**Solution**: Try different ONNX opset version

```bash
python scripts/convert_to_tflite.py \
    --model model.pt \
    --input-shape 1 3 224 224 \
    --opset-version 11  # Try 11, 12, or 13
```

### Error: "Module 'onnx2tf' not found"

**Solution**: Install conversion tools

```bash
pip install onnx2tf
# OR
pip install onnx-tf
```

### Model size too large (>100MB)

**Solution**: Use aggressive quantization

```bash
python scripts/convert_to_tflite.py \
    --model model.pt \
    --input-shape 1 3 224 224 \
    --quantize int8  # Smallest size
```

### Accuracy drops after quantization

**Solution**: Use float16 instead of int8

```bash
--quantize float16  # Better accuracy than int8
```

### "TensorFlow version incompatibility"

**Solution**: Use Python 3.8-3.10 and compatible versions

```bash
# Uninstall
pip uninstall tensorflow onnx onnx2tf

# Reinstall compatible versions
pip install tensorflow==2.12.0
pip install onnx==1.14.0
pip install onnx2tf==1.16.0
```

## Performance Optimization

### 1. Choose Right Quantization

| Type | Size | Speed | Accuracy | Use Case |
|------|------|-------|----------|----------|
| none | 100% | Slow | Best | Testing only |
| dynamic | ~75% | Medium | Good | Recommended |
| float16 | ~50% | Fast | Good | Production |
| int8 | ~25% | Fastest | Fair | Edge devices |

### 2. Model Size Guidelines

- **< 5MB**: Excellent for mobile
- **5-25MB**: Good for mobile
- **25-50MB**: Acceptable, test on budget devices
- **50-100MB**: May cause issues on low-end devices
- **> 100MB**: Requires optimization or WiFi download

### 3. Enable GPU Acceleration

In React Native:

```javascript
import { useTensorflowModel } from 'react-native-fast-tflite';

const model = useTensorflowModel(
  require('./assets/model.tflite'),
  {
    // Enable GPU delegate
    delegate: 'gpu'
  }
);
```

## Complete Example

### 1. Train PyTorch Model

```python
# train_model.py
import torch
import torch.nn as nn

model = YourModel()
# ... training code ...

# Save entire model
torch.save(model, 'wood_classifier.pt')
```

### 2. Convert to TFLite

```bash
python scripts/convert_to_tflite.py \
    --model wood_classifier.pt \
    --input-shape 1 3 224 224 \
    --quantize float16 \
    --output-name wood_classifier \
    --labels non-wood wood
```

### 3. Deploy to React Native

```javascript
// WoodClassifier.js
import React, { useState } from 'react';
import { View, Button, Text } from 'react-native';
import { useTensorflowModel } from 'react-native-fast-tflite';

export default function WoodClassifier() {
  const [result, setResult] = useState(null);

  const model = useTensorflowModel(
    require('./assets/wood_classifier.tflite')
  );

  const labels = ['non-wood', 'wood'];

  async function classify(imagePath) {
    // Load and preprocess image
    const imageData = await loadImage(imagePath);
    const input = preprocessImage(imageData);

    // Run inference
    const output = model.run(input);

    // Get prediction
    const probabilities = Array.from(output);
    const maxIdx = probabilities.indexOf(Math.max(...probabilities));

    setResult({
      class: labels[maxIdx],
      confidence: (probabilities[maxIdx] * 100).toFixed(2) + '%'
    });
  }

  return (
    <View>
      <Button title="Classify" onPress={() => classify('image.jpg')} />
      {result && (
        <Text>
          Prediction: {result.class} ({result.confidence})
        </Text>
      )}
    </View>
  );
}
```

## Resources

### Documentation
- [react-native-fast-tflite GitHub](https://github.com/mrousavy/react-native-fast-tflite)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [ONNX Documentation](https://onnx.ai/onnx/intro/)
- [onnx2tf GitHub](https://github.com/PINTO0309/onnx2tf)

### Tutorials
- [PyTorch to TFLite Conversion](https://medium.com/@amitvermaphd/converting-pytorch-models-to-onnx-and-tflite-for-mobile-apps-bf903d54ba0e)
- [React Native ML Guide 2025](https://javascript.plainenglish.io/react-native-fast-tflite-on-device-machine-learning-guide-2025-906b1a8181b1)
- [From PyTorch to Android](https://deepsense.ai/resource/from-pytorch-to-android-creating-a-quantized-tensorflow-lite-model/)

## Summary

✅ **Conversion Path**: PyTorch → ONNX → TensorFlow → TFLite
✅ **Recommended Tool**: onnx2tf (2025 best practice)
✅ **Quantization**: float16 for production (good size/accuracy balance)
✅ **Input Format**: NHWC (conversion handles this automatically)
✅ **React Native**: Use react-native-fast-tflite v1.6.1
✅ **Testing**: Always test on actual devices before deployment

Good luck with your deployment! 🚀
