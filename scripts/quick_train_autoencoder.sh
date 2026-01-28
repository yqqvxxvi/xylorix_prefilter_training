#!/bin/bash
#
# Quick Start Script for Autoencoder Training
#
# This script trains an autoencoder on end-grain wood features
# using default recommended settings.
#
# Usage: bash scripts/quick_train_autoencoder.sh

set -e  # Exit on error

# Configuration
ENDGRAIN_DIR="/Users/youqing/Documents/training_dataset/batch_proc/endgrain"
WORLD_DIR="/Users/youqing/Documents/training_dataset/batch_proc/world"
OUTPUT_DIR="results/autoencoder_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Autoencoder Quick Start Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  End-grain dir: $ENDGRAIN_DIR"
echo "  World dir:     $WORLD_DIR"
echo "  Output dir:    $OUTPUT_DIR"
echo ""
echo "=========================================="
echo ""

# Check if directories exist
if [ ! -d "$ENDGRAIN_DIR" ]; then
    echo "Error: End-grain directory not found: $ENDGRAIN_DIR"
    echo "Please update ENDGRAIN_DIR in this script"
    exit 1
fi

if [ ! -d "$WORLD_DIR" ]; then
    echo "Error: World directory not found: $WORLD_DIR"
    echo "Please update WORLD_DIR in this script"
    exit 1
fi

# Detect device
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    BATCH_SIZE=64
    echo "CUDA detected! Using GPU acceleration"
elif python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE="mps"
    BATCH_SIZE=32
    echo "MPS (Apple Silicon) detected! Using GPU acceleration"
else
    DEVICE="cpu"
    BATCH_SIZE=16
    echo "Using CPU (training will be slower)"
fi
echo ""

# Train autoencoder
echo "Starting training..."
echo ""

python scripts/train_autoencoder.py \
    --endgrain_dir "$ENDGRAIN_DIR" \
    --world_dir "$WORLD_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_type standard \
    --latent_dim 16 \
    --hidden_dims 128 64 32 \
    --epochs 100 \
    --batch_size $BATCH_SIZE \
    --learning_rate 0.001 \
    --val_split 0.2 \
    --device $DEVICE \
    --normalize_features \
    --cache_features \
    --precompute \
    --num_workers 4 \
    --seed 42

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To visualize the latent space, run:"
echo ""
echo "  python scripts/visualize_latent_space.py \\"
echo "      --checkpoint $OUTPUT_DIR/checkpoints/best_model.pth \\"
echo "      --endgrain_dir $ENDGRAIN_DIR \\"
echo "      --world_dir $WORLD_DIR \\"
echo "      --output_dir $OUTPUT_DIR/latent_viz \\"
echo "      --method all"
echo ""
echo "=========================================="
