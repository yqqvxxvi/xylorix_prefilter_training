"""
Comprehensive feature extraction pipeline for end-grain wood detection

This module implements a complete feature extraction system designed to
distinguish end-grain wood surfaces from non-wood objects (PCBs, furniture,
rough surfaces, etc.) in a mobile computer vision application.

Features extracted:
A. Gradient Orientation Histogram (HOG-like)
B. Gabor Filter Bank
C. Local Binary Pattern (LBP)
D. 2D FFT Analysis
E. Wavelet Features
F. Edge Analysis
G. Blob Detection (Vessel/Pore Detection)
H. Hough Circle Transform
I. GLCM Features
J. First-Order Statistics
"""

import numpy as np
import cv2
from scipy import ndimage, fftpack
from scipy.stats import skew, kurtosis, entropy
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, blob_log
from skimage.transform import hough_circle, hough_circle_peaks
import torch
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class EndGrainFeatureExtractor:
    """
    Comprehensive feature extraction for end-grain wood detection

    This class extracts ~265 features designed to characterize end-grain wood
    texture patterns including radial growth rings, linear vessel patterns,
    porous structures with dark circular features, and periodic textures.

    Example:
        >>> extractor = EndGrainFeatureExtractor()
        >>> features, names = extractor.extract_all_features(image)
        >>> print(f"Extracted {len(features)} features")
    """

    def __init__(self, image_size: int = 224, normalize: bool = True):
        """
        Initialize the feature extractor

        Args:
            image_size: Expected image size (default: 224)
            normalize: Whether to apply z-score normalization to features
        """
        self.image_size = image_size
        self.normalize = normalize

        # Feature A: Gradient orientation histogram parameters
        self.n_orientation_bins = 16  # 16 bins covering 0-360°

        # Feature B: Gabor filter bank parameters
        self.gabor_frequencies = [0.05, 0.1, 0.2]  # Low, medium, high frequency
        self.gabor_orientations = [0, 45, 90, 135]  # 4 orientations in degrees
        self.gabor_kernels = self._build_gabor_kernels()

        # Feature C: LBP parameters
        self.lbp_radii = [1, 2, 3]  # Multi-scale analysis
        self.lbp_n_points = 8

        # Feature E: Wavelet parameters
        self.wavelet_type = 'db4'  # Daubechies 4
        self.wavelet_levels = 2

        # Feature G: Blob detection parameters
        self.blob_min_sigma = 1
        self.blob_max_sigma = 10
        self.blob_threshold = 0.1

        # Feature H: Hough circle parameters
        self.hough_min_radius = 10
        self.hough_max_radius = 100

        # Feature I: GLCM parameters
        self.glcm_distances = [1, 2, 5]
        self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        # Normalization statistics (computed on first batch if normalize=True)
        self.feature_mean = None
        self.feature_std = None

    def _build_gabor_kernels(self) -> List[np.ndarray]:
        """Build Gabor filter bank"""
        kernels = []
        ksize = 31  # Kernel size
        sigma = 5.0  # Gaussian envelope

        for freq in self.gabor_frequencies:
            for theta_deg in self.gabor_orientations:
                theta = np.deg2rad(theta_deg)
                # wavelength from frequency
                wavelength = 1.0 / freq if freq > 0 else 20.0

                kernel = cv2.getGaborKernel(
                    (ksize, ksize),
                    sigma,
                    theta,
                    wavelength,
                    0.5,  # gamma (aspect ratio)
                    0,    # psi (phase offset)
                    ktype=cv2.CV_32F
                )
                kernel /= 1.5 * kernel.sum()
                kernels.append(kernel)

        return kernels

    def _preprocess_image(self, image: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert input to grayscale and RGB numpy arrays

        Args:
            image: Input image (H, W, 3) numpy or (B, 3, H, W) torch tensor

        Returns:
            Tuple of (rgb_image, gray_image) as numpy arrays
        """
        # Handle torch tensors
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:  # Batch tensor (B, C, H, W)
                # Take first image from batch for single image processing
                image = image[0]
            # Convert (C, H, W) to (H, W, C)
            image = image.permute(1, 2, 0).cpu().numpy()

        # Ensure numpy array
        image = np.asarray(image)

        # Handle different formats
        if len(image.shape) == 2:
            # Already grayscale
            gray = image
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            # RGB image
            rgb = image
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # Ensure uint8
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)

        return rgb, gray

    # =========================================================================
    # FEATURE A: Gradient Orientation Histogram (HOG-like)
    # =========================================================================

    def extract_gradient_orientation_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Extract gradient orientation histogram features

        Computes gradient magnitudes and orientations using Sobel filters,
        then creates a histogram of orientations weighted by magnitudes.

        Expected behavior: End-grain shows dominant radial/circular orientations

        Args:
            image: Grayscale image (H, W)

        Returns:
            Orientation distribution vector (n_bins,)
        """
        _, gray = self._preprocess_image(image)

        # Compute gradients using Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Compute magnitude and orientation
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)  # Range: [-pi, pi]

        # Convert to [0, 2*pi]
        orientation = (orientation + np.pi) % (2 * np.pi)

        # Create weighted histogram
        bins = np.linspace(0, 2*np.pi, self.n_orientation_bins + 1)
        hist, _ = np.histogram(orientation, bins=bins, weights=magnitude)

        # Normalize
        hist = hist / (hist.sum() + 1e-8)

        return hist.astype(np.float32)

    # =========================================================================
    # FEATURE B: Gabor Filter Bank
    # =========================================================================

    def extract_gabor_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Gabor filter bank features

        Applies pre-built Gabor filters at multiple frequencies and orientations,
        computing mean and std of responses for each filter.

        Expected behavior: End-grain shows strong responses at specific orientations
        matching vessel directions

        Args:
            image: Grayscale image (H, W)

        Returns:
            Gabor feature vector (n_frequencies × n_orientations × 2,)
            = 3 × 4 × 2 = 24 features
        """
        _, gray = self._preprocess_image(image)
        gray_float = gray.astype(np.float32) / 255.0

        features = []
        for kernel in self.gabor_kernels:
            # Apply filter
            filtered = cv2.filter2D(gray_float, cv2.CV_32F, kernel)

            # Compute statistics
            features.append(np.mean(np.abs(filtered)))
            features.append(np.std(filtered))

        return np.array(features, dtype=np.float32)

    # =========================================================================
    # FEATURE C: Local Binary Pattern (LBP)
    # =========================================================================

    def extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale uniform rotation-invariant LBP features

        Computes LBP at multiple radii (1, 2, 3) to capture texture at
        different scales. Uses uniform patterns to reduce dimensionality.

        Expected behavior: End-grain has characteristic repetitive micro-texture patterns

        Args:
            image: Grayscale image (H, W)

        Returns:
            Concatenated LBP histograms for all scales (~177 features)
            59 uniform patterns × 3 scales = 177 features
        """
        _, gray = self._preprocess_image(image)

        all_histograms = []

        for radius in self.lbp_radii:
            # Compute LBP with uniform rotation-invariant patterns
            lbp = local_binary_pattern(
                gray,
                P=self.lbp_n_points,
                R=radius,
                method='uniform'
            )

            # Compute histogram
            # Uniform LBP with 8 points has 59 patterns (58 uniform + 1 non-uniform)
            n_bins = self.lbp_n_points * (self.lbp_n_points - 1) + 3
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

            # Normalize
            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-8)

            all_histograms.append(hist)

        return np.concatenate(all_histograms)

    # =========================================================================
    # FEATURE D: 2D FFT Analysis
    # =========================================================================

    def extract_fft_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract frequency domain features using 2D FFT

        Computes radial frequency profile and energy distribution in different
        frequency bands. Also computes spectrum entropy.

        Expected behavior: End-grain shows periodic structures with characteristic
        frequency peaks

        Args:
            image: Grayscale image (H, W)

        Returns:
            FFT feature vector (5 features):
            [low_freq_energy, mid_freq_energy, high_freq_energy,
             peak_freq_radius, spectrum_entropy]
        """
        _, gray = self._preprocess_image(image)

        # Apply Hann window to reduce edge effects
        h, w = gray.shape
        window = np.outer(np.hanning(h), np.hanning(w))
        gray_windowed = gray.astype(np.float32) * window

        # Compute 2D FFT
        fft = fftpack.fft2(gray_windowed)
        fft_shift = fftpack.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Get center coordinates
        cy, cx = h // 2, w // 2

        # Create radial distance map
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_radius = min(cy, cx)

        # Frequency band energies
        # Low: 0-20%, Mid: 20-60%, High: 60-100% of max radius
        low_mask = r < (0.2 * max_radius)
        mid_mask = (r >= 0.2 * max_radius) & (r < 0.6 * max_radius)
        high_mask = r >= 0.6 * max_radius

        total_energy = magnitude.sum()
        low_energy = magnitude[low_mask].sum() / total_energy
        mid_energy = magnitude[mid_mask].sum() / total_energy
        high_energy = magnitude[high_mask].sum() / total_energy

        # Peak frequency (excluding DC component)
        magnitude_no_dc = magnitude.copy()
        magnitude_no_dc[cy-5:cy+6, cx-5:cx+6] = 0
        peak_idx = np.unravel_index(np.argmax(magnitude_no_dc), magnitude.shape)
        peak_radius = np.sqrt((peak_idx[0] - cy)**2 + (peak_idx[1] - cx)**2)

        # Spectrum entropy
        magnitude_flat = magnitude.ravel()
        magnitude_flat = magnitude_flat / (magnitude_flat.sum() + 1e-8)
        spectrum_entropy = entropy(magnitude_flat + 1e-8)

        features = np.array([
            low_energy,
            mid_energy,
            high_energy,
            peak_radius / max_radius,  # Normalized
            spectrum_entropy
        ], dtype=np.float32)

        return features

    # =========================================================================
    # FEATURE E: Wavelet Features
    # =========================================================================

    def extract_wavelet_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract wavelet decomposition features

        Performs 2-level wavelet decomposition and computes energy in each
        subband (LL, LH, HL, HH at each level).

        Expected behavior: End-grain has specific multi-scale texture energy distribution

        Args:
            image: Grayscale image (H, W)

        Returns:
            Wavelet energy vector (12 features for 2 levels)
            Level 1: LL, LH, HL, HH + Level 2: LL, LH, HL, HH = 8 subbands
            Each subband: mean, std = 12 features
        """
        _, gray = self._preprocess_image(image)

        # Use PyWavelets if available, otherwise use simple approximation
        try:
            import pywt

            # Perform 2-level decomposition
            coeffs = pywt.wavedec2(gray, self.wavelet_type, level=self.wavelet_levels)

            features = []

            # Extract energy from each subband
            # coeffs structure: [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]
            for i, coeff_level in enumerate(coeffs):
                if i == 0:
                    # Approximation coefficients (LL)
                    energy = np.mean(coeff_level**2)
                    features.append(energy)
                else:
                    # Detail coefficients (LH, HL, HH)
                    cH, cV, cD = coeff_level
                    features.append(np.mean(cH**2))  # LH
                    features.append(np.mean(cV**2))  # HL
                    features.append(np.mean(cD**2))  # HH

            return np.array(features, dtype=np.float32)

        except ImportError:
            # Fallback: simple multi-scale energy using Gaussian pyramid
            features = []
            current = gray.astype(np.float32)

            for level in range(self.wavelet_levels):
                # Downsample
                downsampled = cv2.pyrDown(current)

                # Upsample back
                upsampled = cv2.pyrUp(downsampled, dstsize=(current.shape[1], current.shape[0]))

                # Detail (high-frequency component)
                detail = current - upsampled

                # Energy in detail
                features.append(np.mean(detail**2))
                features.append(np.std(detail))

                current = downsampled

            # Approximation energy at final level
            features.append(np.mean(current**2))
            features.append(np.std(current))

            # Pad to 12 features
            while len(features) < 12:
                features.append(0.0)

            return np.array(features[:12], dtype=np.float32)

    # =========================================================================
    # FEATURE F: Edge Analysis
    # =========================================================================

    def extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract edge detection and analysis features

        Applies Canny edge detection and computes edge density, orientation
        distribution, and mean edge strength.

        Expected behavior: End-grain has high edge density from vessel boundaries

        Args:
            image: Grayscale image (H, W)

        Returns:
            Edge feature vector (10 features):
            [edge_density, edge_mean_strength, edge_orientation_histogram (8 bins)]
        """
        _, gray = self._preprocess_image(image)

        # Compute gradients for edge strength
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Adaptive Canny thresholds based on gradient statistics
        median_mag = np.median(magnitude)
        lower = int(max(0, 0.66 * median_mag))
        upper = int(min(255, 1.33 * median_mag))

        # Canny edge detection
        edges = cv2.Canny(gray, lower, upper)

        # Edge density
        edge_density = edges.sum() / (255 * edges.size)

        # Mean edge strength (magnitude at edge pixels)
        edge_mask = edges > 0
        if edge_mask.any():
            mean_strength = magnitude[edge_mask].mean() / 255.0
        else:
            mean_strength = 0.0

        # Edge orientation distribution (8 bins)
        orientation = np.arctan2(grad_y, grad_x)
        orientation = (orientation + np.pi) % (2 * np.pi)

        # Only consider orientations at edge pixels
        edge_orientations = orientation[edge_mask]

        if len(edge_orientations) > 0:
            bins = np.linspace(0, 2*np.pi, 9)
            hist, _ = np.histogram(edge_orientations, bins=bins)
            hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        else:
            hist = np.zeros(8, dtype=np.float32)

        features = np.concatenate([
            [edge_density, mean_strength],
            hist
        ])

        return features.astype(np.float32)

    # =========================================================================
    # FEATURE G: Blob Detection (Vessel/Pore Detection)
    # =========================================================================

    def extract_blob_features(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and analyze circular blobs (vessels/pores)

        Uses Laplacian of Gaussian (LoG) blob detector to find dark circular
        features characteristic of wood pores.

        Expected behavior: End-grain has numerous regularly-spaced circular pores

        Args:
            image: Grayscale image (H, W)

        Returns:
            Blob feature vector (4 features):
            [num_blobs, mean_blob_size, std_blob_size, spatial_regularity]
        """
        _, gray = self._preprocess_image(image)

        # Invert for blob detection (detect dark regions)
        gray_inv = 255 - gray

        # Detect blobs using LoG
        blobs = blob_log(
            gray_inv,
            min_sigma=self.blob_min_sigma,
            max_sigma=self.blob_max_sigma,
            threshold=self.blob_threshold
        )

        num_blobs = len(blobs)

        if num_blobs == 0:
            return np.zeros(4, dtype=np.float32)

        # Blob sizes (radius = sigma * sqrt(2))
        blob_sizes = blobs[:, 2] * np.sqrt(2)
        mean_size = blob_sizes.mean()
        std_size = blob_sizes.std()

        # Spatial regularity: coefficient of variation of inter-blob distances
        if num_blobs >= 2:
            centroids = blobs[:, :2]
            distances = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    distances.append(dist)

            if len(distances) > 0:
                distances = np.array(distances)
                # Regularity score: lower CV = more regular spacing
                cv = distances.std() / (distances.mean() + 1e-8)
                regularity = 1.0 / (1.0 + cv)  # Normalize to [0, 1]
            else:
                regularity = 0.0
        else:
            regularity = 0.0

        features = np.array([
            num_blobs / 100.0,  # Normalize
            mean_size,
            std_size,
            regularity
        ], dtype=np.float32)

        return features

    # =========================================================================
    # FEATURE H: Hough Circle Transform
    # =========================================================================

    def extract_hough_circle_features(self, image: np.ndarray) -> np.ndarray:
        """
        Detect circular/curved structures using Hough transform

        Detects growth rings and circular structures characteristic of end-grain.

        Expected behavior: End-grain may show growth ring curvature

        Args:
            image: Grayscale image (H, W)

        Returns:
            Hough circle feature vector (3 features):
            [num_circles, mean_radius, mean_confidence]
        """
        _, gray = self._preprocess_image(image)

        # Edge detection for Hough transform
        edges = cv2.Canny(gray, 50, 150)

        # Search for circles in a range of radii
        radii = np.arange(self.hough_min_radius, self.hough_max_radius, 10)

        try:
            hough_res = hough_circle(edges, radii)

            # Find peaks
            accums, cx, cy, radii_found = hough_circle_peaks(
                hough_res, radii,
                min_xdistance=20,
                min_ydistance=20,
                threshold=0.4 * hough_res.max(),
                num_peaks=10
            )

            num_circles = len(accums)

            if num_circles == 0:
                return np.zeros(3, dtype=np.float32)

            mean_radius = radii_found.mean() / self.hough_max_radius  # Normalize
            mean_confidence = accums.mean() / (hough_res.max() + 1e-8)

            features = np.array([
                num_circles / 10.0,  # Normalize
                mean_radius,
                mean_confidence
            ], dtype=np.float32)

        except Exception:
            # If Hough transform fails, return zeros
            features = np.zeros(3, dtype=np.float32)

        return features

    # =========================================================================
    # FEATURE I: GLCM Features
    # =========================================================================

    def extract_glcm_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Gray-Level Co-occurrence Matrix (GLCM) features

        Computes texture features including contrast, homogeneity, energy,
        correlation, dissimilarity, and entropy.

        Expected behavior: End-grain has characteristic texture uniformity
        and contrast patterns

        Args:
            image: Grayscale image (H, W)

        Returns:
            GLCM feature vector (6 features, averaged across distances/angles)
        """
        _, gray = self._preprocess_image(image)

        # Reduce gray levels for computational efficiency
        gray_reduced = (gray // 32).astype(np.uint8)  # 256 -> 8 levels

        # Compute GLCM
        glcm = graycomatrix(
            gray_reduced,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=8,
            symmetric=True,
            normed=True
        )

        # Compute properties and average across all distances and angles
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()

        # Entropy (manual computation)
        glcm_mean = glcm.mean(axis=(2, 3))  # Average across distances and angles
        glcm_entropy = -np.sum(glcm_mean * np.log(glcm_mean + 1e-8))

        features = np.array([
            contrast,
            homogeneity,
            energy,
            correlation,
            dissimilarity,
            glcm_entropy
        ], dtype=np.float32)

        return features

    # =========================================================================
    # FEATURE J: First-Order Statistics
    # =========================================================================

    def extract_statistical_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract first-order statistical features

        Computes basic intensity statistics including mean, std, skewness,
        kurtosis, entropy, and percentiles.

        Expected behavior: End-grain has specific intensity distributions

        Args:
            image: Grayscale image (H, W)

        Returns:
            Statistical feature vector (8 features):
            [mean, std, skewness, kurtosis, entropy, p25, p50, p75]
        """
        _, gray = self._preprocess_image(image)

        # Flatten and normalize to [0, 1]
        pixels = gray.ravel().astype(np.float32) / 255.0

        # Compute statistics
        mean_val = pixels.mean()
        std_val = pixels.std()
        skewness = skew(pixels)
        kurt = kurtosis(pixels)

        # Entropy
        hist, _ = np.histogram(pixels, bins=256, range=(0, 1))
        hist = hist.astype(np.float32) / hist.sum()
        ent = entropy(hist + 1e-8)

        # Percentiles
        p25 = np.percentile(pixels, 25)
        p50 = np.percentile(pixels, 50)
        p75 = np.percentile(pixels, 75)

        features = np.array([
            mean_val,
            std_val,
            skewness,
            kurt,
            ent,
            p25,
            p50,
            p75
        ], dtype=np.float32)

        return features

    # =========================================================================
    # FEATURE K: Complete Feature Extraction Pipeline
    # =========================================================================

    def extract_all_features(
        self,
        image: Union[np.ndarray, torch.Tensor, str, Path]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all features from image(s)

        This is the main entry point for feature extraction. Supports both
        single images and batches.

        Args:
            image: Input image as:
                - (H, W, 3) numpy array (single RGB image)
                - (B, H, W, 3) numpy array (batch of RGB images)
                - (3, H, W) torch tensor (single image)
                - (B, 3, H, W) torch tensor (batch)
                - str/Path to image file

        Returns:
            Tuple of (features, feature_names):
                - features: (n_features,) for single image or (B, n_features) for batch
                - feature_names: List of feature names
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Failed to load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Handle batch processing
        is_batch = False
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:
                is_batch = True
                batch_size = image.shape[0]
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 4:
                is_batch = True
                batch_size = image.shape[0]

        if is_batch:
            # Process each image in batch
            all_features = []
            for i in range(batch_size):
                if isinstance(image, torch.Tensor):
                    img = image[i]
                else:
                    img = image[i]
                feats, names = self._extract_single_image_features(img)
                all_features.append(feats)

            features = np.stack(all_features, axis=0)

            # Normalize if requested
            if self.normalize:
                features = self._normalize_features(features)

            return features, names
        else:
            # Single image
            features, names = self._extract_single_image_features(image)

            # Normalize if requested
            if self.normalize:
                features = self._normalize_features(features.reshape(1, -1)).ravel()

            return features, names

    def _extract_single_image_features(self, image: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, List[str]]:
        """Extract features from a single image"""
        feature_dict = {}

        # Feature A: Gradient Orientation Histogram
        grad_hist = self.extract_gradient_orientation_histogram(image)
        for i, val in enumerate(grad_hist):
            feature_dict[f'grad_orient_bin_{i:02d}'] = val

        # Feature B: Gabor Features
        gabor_feats = self.extract_gabor_features(image)
        idx = 0
        for f_idx in range(len(self.gabor_frequencies)):
            for o_idx in range(len(self.gabor_orientations)):
                feature_dict[f'gabor_f{f_idx}_o{o_idx}_mean'] = gabor_feats[idx]
                feature_dict[f'gabor_f{f_idx}_o{o_idx}_std'] = gabor_feats[idx + 1]
                idx += 2

        # Feature C: LBP Features
        lbp_feats = self.extract_lbp_features(image)
        for i, val in enumerate(lbp_feats):
            feature_dict[f'lbp_bin_{i:03d}'] = val

        # Feature D: FFT Features
        fft_feats = self.extract_fft_features(image)
        fft_names = ['fft_low_energy', 'fft_mid_energy', 'fft_high_energy',
                     'fft_peak_radius', 'fft_entropy']
        for name, val in zip(fft_names, fft_feats):
            feature_dict[name] = val

        # Feature E: Wavelet Features
        wavelet_feats = self.extract_wavelet_features(image)
        for i, val in enumerate(wavelet_feats):
            feature_dict[f'wavelet_energy_{i:02d}'] = val

        # Feature F: Edge Features
        edge_feats = self.extract_edge_features(image)
        feature_dict['edge_density'] = edge_feats[0]
        feature_dict['edge_mean_strength'] = edge_feats[1]
        for i, val in enumerate(edge_feats[2:]):
            feature_dict[f'edge_orient_bin_{i}'] = val

        # Feature G: Blob Features
        blob_feats = self.extract_blob_features(image)
        blob_names = ['blob_count', 'blob_mean_size', 'blob_std_size', 'blob_regularity']
        for name, val in zip(blob_names, blob_feats):
            feature_dict[name] = val

        # Feature H: Hough Circle Features
        hough_feats = self.extract_hough_circle_features(image)
        hough_names = ['hough_num_circles', 'hough_mean_radius', 'hough_mean_confidence']
        for name, val in zip(hough_names, hough_feats):
            feature_dict[name] = val

        # Feature I: GLCM Features
        glcm_feats = self.extract_glcm_features(image)
        glcm_names = ['glcm_contrast', 'glcm_homogeneity', 'glcm_energy',
                      'glcm_correlation', 'glcm_dissimilarity', 'glcm_entropy']
        for name, val in zip(glcm_names, glcm_feats):
            feature_dict[name] = val

        # Feature J: Statistical Features
        stat_feats = self.extract_statistical_features(image)
        stat_names = ['stat_mean', 'stat_std', 'stat_skewness', 'stat_kurtosis',
                     'stat_entropy', 'stat_p25', 'stat_p50', 'stat_p75']
        for name, val in zip(stat_names, stat_feats):
            feature_dict[name] = val

        # Convert to arrays
        feature_names = list(feature_dict.keys())
        features = np.array([feature_dict[name] for name in feature_names], dtype=np.float32)

        return features, feature_names

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to features

        Args:
            features: (B, n_features) or (n_features,) array

        Returns:
            Normalized features
        """
        if self.feature_mean is None or self.feature_std is None:
            # Compute statistics from current batch
            if features.ndim == 1:
                self.feature_mean = features.copy()
                self.feature_std = np.ones_like(features)
            else:
                self.feature_mean = features.mean(axis=0)
                self.feature_std = features.std(axis=0) + 1e-8

        # Apply normalization
        features_norm = (features - self.feature_mean) / self.feature_std

        return features_norm

    def get_feature_count(self) -> int:
        """Get total number of features extracted"""
        # Create dummy image to count features
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        features, _ = self._extract_single_image_features(dummy)
        return len(features)
