"""
Texture-based feature extraction for wood microscopy classification

This module implements advanced texture features:
1. Fuzzy Local Binary Pattern (LBP) - PRIMARY
2. Gabor filters (directional texture)
3. Gray-Level Co-occurrence Matrix (GLCM/Haralick features)
4. Frequency domain features (FFT-based)
5. Enhanced blob distribution analysis
"""

import cv2
import numpy as np
from scipy import ndimage, fftpack
from scipy.spatial.distance import pdist, squareform
from skimage.feature import graycomatrix, graycoprops
import math
from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path


# ============================================================================
# 1. FUZZY LOCAL BINARY PATTERN (PRIMARY FEATURE)
# ============================================================================

class FuzzyLBP:
    """
    Fuzzy Local Binary Pattern for robust texture analysis

    Benefits for microscopy:
    - Handles noise and varying contrast better than standard LBP
    - Fuzzy membership functions smooth transitions
    - Rotation-invariant option for orientation-independent features
    """

    def __init__(self, radius: int = 2, n_points: int = 16,
                 fuzzy_threshold: float = 3.0, rotation_invariant: bool = True):
        self.radius = radius
        self.n_points = n_points
        self.fuzzy_threshold = fuzzy_threshold
        self.rotation_invariant = rotation_invariant
        self._compute_sampling_positions()

    def _compute_sampling_positions(self):
        """Compute circular sampling positions"""
        angles = 2 * np.pi * np.arange(self.n_points) / self.n_points
        self.offsets_x = -self.radius * np.sin(angles)
        self.offsets_y = self.radius * np.cos(angles)

    def _rotation_invariant_mapping(self, codes: np.ndarray) -> np.ndarray:
        """Apply rotation invariant mapping to LBP codes"""
        h, w = codes.shape
        ri_codes = np.zeros_like(codes)

        for i in range(h):
            for j in range(w):
                code = int(codes[i, j])
                min_code = code
                temp_code = code

                for _ in range(self.n_points):
                    temp_code = ((temp_code << 1) | (temp_code >> (self.n_points - 1))) & ((1 << self.n_points) - 1)
                    min_code = min(min_code, temp_code)

                ri_codes[i, j] = min_code

        return ri_codes

    def extract(self, image: np.ndarray,
                roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Extract fuzzy LBP histogram from image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        pad_size = self.radius + 1
        padded = np.pad(gray, pad_size, mode='reflect')

        h, w = gray.shape
        fuzzy_codes = np.zeros((h, w), dtype=np.float32)

        angles = 2 * np.pi * np.arange(self.n_points) / self.n_points
        x_coords = self.radius * np.cos(angles)
        y_coords = -self.radius * np.sin(angles)

        for i in range(h):
            for j in range(w):
                center_i = i + pad_size
                center_j = j + pad_size
                center = padded[center_i, center_j]

                fuzzy_pattern = 0.0
                weight_sum = 0.0

                for p in range(self.n_points):
                    y_p = center_i + y_coords[p]
                    x_p = center_j + x_coords[p]

                    y0 = int(np.floor(y_p))
                    y1 = min(y0 + 1, padded.shape[0] - 1)
                    x0 = int(np.floor(x_p))
                    x1 = min(x0 + 1, padded.shape[1] - 1)

                    wy = y_p - y0
                    wx = x_p - x0

                    neighbor = (1 - wy) * (1 - wx) * padded[y0, x0] + \
                              (1 - wy) * wx * padded[y0, x1] + \
                              wy * (1 - wx) * padded[y1, x0] + \
                              wy * wx * padded[y1, x1]

                    diff = neighbor - center

                    if abs(diff) <= self.fuzzy_threshold:
                        membership = 1.0 - abs(diff) / self.fuzzy_threshold
                    else:
                        membership = 0.0

                    fuzzy_pattern += membership * (2 ** p)
                    weight_sum += membership

                if weight_sum > 0:
                    fuzzy_codes[i, j] = fuzzy_pattern / weight_sum
                else:
                    fuzzy_codes[i, j] = 0

        if self.rotation_invariant:
            fuzzy_codes = self._rotation_invariant_mapping(fuzzy_codes)

        n_bins = self.n_points + 2 if self.rotation_invariant else 2 ** self.n_points
        fuzzy_codes_clipped = np.clip(fuzzy_codes.flatten(), 0, n_bins - 1)
        hist, _ = np.histogram(fuzzy_codes_clipped, bins=n_bins, range=(0, n_bins))

        hist = hist.astype(np.float32)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum

        return hist

    def extract_features(self, image: np.ndarray,
                        roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """Extract fuzzy LBP features as named dictionary"""
        hist = self.extract(image, roi)
        features = {}

        for i, val in enumerate(hist):
            features[f'flbp_bin_{i:02d}'] = float(val)

        features['flbp_uniformity'] = float(np.sum(hist ** 2))

        hist_nonzero = hist[hist > 0]
        features['flbp_entropy'] = float(-np.sum(hist_nonzero * np.log2(hist_nonzero)))

        mean = np.sum(np.arange(len(hist)) * hist)
        features['flbp_contrast'] = float(np.sum(((np.arange(len(hist)) - mean) ** 2) * hist))

        return features


# ============================================================================
# 2. GABOR FILTERS (Directional Texture Analysis)
# ============================================================================

class GaborFeatures:
    """Gabor filter bank for directional texture analysis"""

    def __init__(self, frequencies: List[float] = [0.05, 0.1, 0.2],
                 orientations: int = 8, ksize: int = 31, sigma: float = 3.0):
        self.frequencies = frequencies
        self.orientations = orientations
        self.ksize = ksize
        self.sigma = sigma
        self.filters = self._build_filter_bank()

    def _build_filter_bank(self) -> List[np.ndarray]:
        """Build Gabor filter bank"""
        filters = []
        for freq in self.frequencies:
            for theta in np.arange(0, np.pi, np.pi / self.orientations):
                kern = cv2.getGaborKernel(
                    (self.ksize, self.ksize),
                    self.sigma,
                    theta,
                    2 * np.pi / freq if freq > 0 else 10.0,
                    0.5,
                    0,
                    ktype=cv2.CV_32F
                )
                kern /= 1.5 * kern.sum()
                filters.append(kern)
        return filters

    def extract_features(self, image: np.ndarray,
                        roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """Extract Gabor features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        gray = gray.astype(np.float32) / 255.0

        features = {}
        responses = []

        idx = 0
        for f_idx, freq in enumerate(self.frequencies):
            for o_idx in range(self.orientations):
                filtered = cv2.filter2D(gray, cv2.CV_32F, self.filters[idx])

                features[f'gabor_f{f_idx}_o{o_idx}_mean'] = float(np.mean(np.abs(filtered)))
                features[f'gabor_f{f_idx}_o{o_idx}_std'] = float(np.std(filtered))

                responses.append(np.mean(np.abs(filtered)))
                idx += 1

        responses = np.array(responses).reshape(len(self.frequencies), self.orientations)

        for f_idx in range(len(self.frequencies)):
            dominant_orient = np.argmax(responses[f_idx, :])
            features[f'gabor_f{f_idx}_dominant_orient'] = float(dominant_orient)

        orientation_energy = np.sum(responses, axis=0)
        if np.sum(orientation_energy) > 0:
            orientation_energy /= np.sum(orientation_energy)
            anisotropy = 1.0 - np.sum(orientation_energy ** 2) * self.orientations
            features['gabor_anisotropy'] = float(anisotropy)
        else:
            features['gabor_anisotropy'] = 0.0

        return features


# ============================================================================
# 3. GRAY-LEVEL CO-OCCURRENCE MATRIX (GLCM/Haralick Features)
# ============================================================================

class GLCMFeatures:
    """GLCM-based Haralick texture features"""

    def __init__(self, distances: List[int] = [1, 3, 5],
                 angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                 levels: int = 256):
        self.distances = distances
        self.angles = angles
        self.levels = levels

    def extract_features(self, image: np.ndarray,
                        roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """Extract GLCM Haralick features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        features = {}

        glcm = graycomatrix(
            gray,
            distances=self.distances,
            angles=self.angles,
            levels=self.levels,
            symmetric=True,
            normed=True
        )

        properties = ['contrast', 'dissimilarity', 'homogeneity',
                     'energy', 'correlation', 'ASM']

        for d_idx, dist in enumerate(self.distances):
            for prop in properties:
                values = graycoprops(glcm, prop)[:, d_idx]
                features[f'glcm_d{dist}_{prop}'] = float(np.mean(values))

        return features


# ============================================================================
# 4. FREQUENCY DOMAIN FEATURES
# ============================================================================

class FrequencyFeatures:
    """Frequency domain features using FFT"""

    def __init__(self, n_radial_bins: int = 10, n_angular_bins: int = 8):
        self.n_radial_bins = n_radial_bins
        self.n_angular_bins = n_angular_bins

    def extract_features(self, image: np.ndarray,
                        roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """Extract frequency domain features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        window = np.outer(
            np.hanning(gray.shape[0]),
            np.hanning(gray.shape[1])
        )
        gray_windowed = gray.astype(np.float32) * window

        fft = fftpack.fft2(gray_windowed)
        fft_shift = fftpack.fftshift(fft)
        magnitude = np.abs(fft_shift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        features = {}

        dc_component = magnitude[cy, cx]
        total_energy = np.sum(magnitude)
        features['freq_dc_ratio'] = float(dc_component / total_energy if total_energy > 0 else 0)

        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        theta = np.arctan2(y - cy, x - cx)

        max_radius = min(cy, cx)
        radial_bins = np.linspace(0, max_radius, self.n_radial_bins + 1)

        for i in range(self.n_radial_bins):
            mask = (r >= radial_bins[i]) & (r < radial_bins[i + 1])
            if np.any(mask):
                features[f'freq_radial_{i}'] = float(np.mean(magnitude[mask]))
            else:
                features[f'freq_radial_{i}'] = 0.0

        angular_bins = np.linspace(-np.pi, np.pi, self.n_angular_bins + 1)

        for i in range(self.n_angular_bins):
            mask = (theta >= angular_bins[i]) & (theta < angular_bins[i + 1])
            mask &= (r > 5)
            if np.any(mask):
                features[f'freq_angular_{i}'] = float(np.mean(magnitude[mask]))
            else:
                features[f'freq_angular_{i}'] = 0.0

        magnitude_no_dc = magnitude.copy()
        magnitude_no_dc[cy-5:cy+5, cx-5:cx+5] = 0

        peak_idx = np.unravel_index(np.argmax(magnitude_no_dc), magnitude_no_dc.shape)
        features['freq_peak_magnitude'] = float(magnitude_no_dc[peak_idx])
        features['freq_peak_radius'] = float(np.sqrt((peak_idx[0] - cy)**2 + (peak_idx[1] - cx)**2))

        return features


# ============================================================================
# 5. INTEGRATED FEATURE EXTRACTOR
# ============================================================================

class TextureFeatureExtractor:
    """Integrated texture feature extractor combining all methods"""

    def __init__(self,
                 use_fuzzy_lbp: bool = True,
                 use_gabor: bool = True,
                 use_glcm: bool = True,
                 use_frequency: bool = True,
                 roi_center_crop: Optional[float] = 0.8):
        """
        Args:
            use_*: Enable/disable specific feature types
            roi_center_crop: Fraction of image to use (0.8 = center 80%)
        """
        self.use_fuzzy_lbp = use_fuzzy_lbp
        self.use_gabor = use_gabor
        self.use_glcm = use_glcm
        self.use_frequency = use_frequency
        self.roi_center_crop = roi_center_crop

        if self.use_fuzzy_lbp:
            self.flbp = FuzzyLBP(radius=2, n_points=16, fuzzy_threshold=3.0)

        if self.use_gabor:
            self.gabor = GaborFeatures(frequencies=[0.05, 0.1, 0.2], orientations=8)

        if self.use_glcm:
            self.glcm = GLCMFeatures(distances=[1, 3, 5])

        if self.use_frequency:
            self.frequency = FrequencyFeatures(n_radial_bins=10, n_angular_bins=8)

    def _get_roi(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Compute ROI for center cropping"""
        if self.roi_center_crop is None:
            return None

        h, w = image.shape[:2]
        crop_h = int(h * self.roi_center_crop)
        crop_w = int(w * self.roi_center_crop)

        y = (h - crop_h) // 2
        x = (w - crop_w) // 2

        return (x, y, crop_w, crop_h)

    def extract_features(self, image: Union[str, Path, np.ndarray]) -> Dict[str, float]:
        """
        Extract all texture features from image

        Args:
            image: Input image (path or numpy array)

        Returns:
            Dictionary with all texture features
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
        else:
            img = image

        roi = self._get_roi(img)
        all_features = {}

        if self.use_fuzzy_lbp:
            flbp_features = self.flbp.extract_features(img, roi)
            all_features.update(flbp_features)

        if self.use_gabor:
            gabor_features = self.gabor.extract_features(img, roi)
            all_features.update(gabor_features)

        if self.use_glcm:
            glcm_features = self.glcm.extract_features(img, roi)
            all_features.update(glcm_features)

        if self.use_frequency:
            freq_features = self.frequency.extract_features(img, roi)
            all_features.update(freq_features)

        return all_features
