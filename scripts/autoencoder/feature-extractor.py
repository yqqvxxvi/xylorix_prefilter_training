import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.autoencoder import EndGrainFeatureExtractor, FeatureVisualizer

# Initialize extractor
extractor = EndGrainFeatureExtractor(image_size=224, normalize=True)

# Extract features from image
path = '/Users/youqing/Documents/training_dataset/batch_proc/endgrain/WhatsApp Image 2025-12-05 at 12.25.20 (1).jpeg'
features, feature_names = extractor.extract_all_features(path)

print(f"Extracted {len(features)} features")

# Create output directory if it doesn't exist
output_dir = Path('results/feature_img')
output_dir.mkdir(parents=True, exist_ok=True)

# Visualize features
visualizer = FeatureVisualizer(extractor)
visualizer.visualize_all_features(path, save_path='results/feature_img/features.png')

print(f"Visualization saved to: results/feature_img/features.png")