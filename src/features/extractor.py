"""
Unified feature extraction pipeline
Combines quality check (VoL), blob analysis, and texture features
"""

import numpy as np
from typing import Dict, Union, Optional
from pathlib import Path

from .quality import calculate_vol, is_image_clear
from .blob import extract_blob_features, BLOB_FEATURE_NAMES
from .texture import TextureFeatureExtractor


class WoodFeatureExtractor:
    """
    Complete feature extraction pipeline for wood classification

    Workflow:
    1. Quality check (VoL blur detection)
    2. Blob analysis (pore detection)
    3. Texture features (LBP, Gabor, GLCM, FFT)

    Example:
        >>> extractor = WoodFeatureExtractor(use_texture=True)
        >>> features = extractor.extract('image.jpg')
        >>> print(f"Extracted {len(features)} features")
    """

    def __init__(self,
                 vol_threshold: float = 900,
                 threshold_value: int = 40,
                 min_blob_area: float = 3,
                 max_blob_area: float = 40,
                 use_texture: bool = False,
                 texture_params: Optional[Dict] = None,
                 skip_vol_check: bool = False,
                 resize_to: tuple = (320, 320)):
        """
        Args:
            vol_threshold: VoL threshold for blur detection (default 900)
            threshold_value: Binary threshold for blob detection (default 40)
            min_blob_area: Minimum blob area in pixels (default 3)
            max_blob_area: Maximum blob area in pixels (default 40)
            use_texture: Whether to extract texture features
            texture_params: Optional texture feature parameters
            skip_vol_check: If True, extract features even for blurry images (for usability task)
            resize_to: Resize images to (width, height) before blob detection (default 320x320)
        """
        self.vol_threshold = vol_threshold
        self.threshold_value = threshold_value
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.use_texture = use_texture
        self.skip_vol_check = skip_vol_check
        self.resize_to = resize_to

        # Initialize texture extractor if needed
        if self.use_texture:
            texture_params = texture_params or {}
            self.texture_extractor = TextureFeatureExtractor(**texture_params)
        else:
            self.texture_extractor = None

    def extract(self, image_path: Union[str, Path]) -> Dict[str, float]:
        """
        Extract all features from image

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with features:
            - vol_score: VoL sharpness score
            - is_clear: Boolean clarity flag (0 or 1)
            - Blob features (11 features)
            - Texture features (optional, ~100+ features)

        Example:
            >>> features = extractor.extract('wood_sample.jpg')
            >>> if features['is_clear']:
            ...     print(f"Detected {features['num_filtered_blobs']} pores")
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        features = {}

        # 1. Quality check (VoL)
        vol_score = calculate_vol(str(image_path))
        features['vol_score'] = vol_score
        features['is_clear'] = float(vol_score >= self.vol_threshold)

        # If image is blurry, return early
        # Skip feature extraction for blurry images (unless skip_vol_check is True)
        if not features['is_clear'] and not self.skip_vol_check:
            # Pad with zeros for blob features
            for feat_name in BLOB_FEATURE_NAMES:
                features[feat_name] = 0.0
            return features

        # 2. Blob analysis (extract even if blurry when skip_vol_check=True)
        blob_features = extract_blob_features(
            str(image_path),
            threshold_value=self.threshold_value,
            min_blob_area=self.min_blob_area,
            max_blob_area=self.max_blob_area,
            resize_to=self.resize_to
        )
        features.update(blob_features)

        # 3. Texture features (optional)
        if self.use_texture and self.texture_extractor is not None:
            texture_features = self.texture_extractor.extract_features(str(image_path))
            features.update(texture_features)

        return features

    def get_feature_names(self) -> list:
        """
        Get ordered list of all feature names

        Returns:
            List of feature names in order
        """
        feature_names = ['vol_score', 'is_clear'] + BLOB_FEATURE_NAMES

        if self.use_texture and self.texture_extractor is not None:
            # Get texture feature names by extracting from a dummy feature dict
            # This is a bit hacky but ensures consistent ordering
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            import cv2
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                cv2.imwrite(f.name, dummy_img)
                try:
                    texture_feat = self.texture_extractor.extract_features(f.name)
                    texture_names = list(texture_feat.keys())
                    feature_names.extend(texture_names)
                except Exception:
                    pass
                finally:
                    Path(f.name).unlink(missing_ok=True)

        return feature_names
