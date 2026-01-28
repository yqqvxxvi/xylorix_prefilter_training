"""
Visualization utilities for end-grain feature extraction

Provides comprehensive visualization of all extracted features including:
- Gabor filter responses
- FFT power spectrum
- Detected blobs
- Edge detection results
- LBP histograms
- Multi-panel comprehensive visualization
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import fftpack
from skimage.feature import blob_log, local_binary_pattern
from skimage.transform import hough_circle, hough_circle_peaks
from typing import Union, Optional, Tuple
from pathlib import Path

from .feature_extractor import EndGrainFeatureExtractor


class FeatureVisualizer:
    """
    Visualization utilities for end-grain feature extraction

    Example:
        >>> visualizer = FeatureVisualizer()
        >>> visualizer.visualize_all_features(image, save_path='features.png')
    """

    def __init__(self, extractor: Optional[EndGrainFeatureExtractor] = None):
        """
        Initialize visualizer

        Args:
            extractor: Optional feature extractor instance. If None, creates default.
        """
        self.extractor = extractor or EndGrainFeatureExtractor()

    def _load_and_preprocess(self, image: Union[np.ndarray, str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and convert to RGB and grayscale"""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return rgb, gray

    def visualize_gabor_responses(
        self,
        image: Union[np.ndarray, str, Path],
        save_path: Optional[str] = None
    ):
        """
        Visualize all Gabor filter responses in a grid

        Args:
            image: Input image
            save_path: Optional path to save visualization
        """
        rgb, gray = self._load_and_preprocess(image)
        gray_float = gray.astype(np.float32) / 255.0

        n_freq = len(self.extractor.gabor_frequencies)
        n_orient = len(self.extractor.gabor_orientations)

        fig, axes = plt.subplots(n_freq, n_orient, figsize=(12, 9))
        fig.suptitle('Gabor Filter Responses', fontsize=16, fontweight='bold')

        idx = 0
        for f_idx, freq in enumerate(self.extractor.gabor_frequencies):
            for o_idx, orient in enumerate(self.extractor.gabor_orientations):
                kernel = self.extractor.gabor_kernels[idx]
                filtered = cv2.filter2D(gray_float, cv2.CV_32F, kernel)

                ax = axes[f_idx, o_idx] if n_freq > 1 else axes[o_idx]
                im = ax.imshow(filtered, cmap='RdBu_r')
                ax.set_title(f'f={freq:.2f}, θ={orient}°', fontsize=8)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)

                idx += 1

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_fft_spectrum(
        self,
        image: Union[np.ndarray, str, Path],
        save_path: Optional[str] = None
    ):
        """
        Visualize FFT power spectrum with annotations

        Args:
            image: Input image
            save_path: Optional path to save visualization
        """
        rgb, gray = self._load_and_preprocess(image)

        # Apply Hann window
        h, w = gray.shape
        window = np.outer(np.hanning(h), np.hanning(w))
        gray_windowed = gray.astype(np.float32) * window

        # Compute FFT
        fft = fftpack.fft2(gray_windowed)
        fft_shift = fftpack.fftshift(fft)
        magnitude = np.abs(fft_shift)
        magnitude_log = np.log(magnitude + 1)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('FFT Frequency Analysis', fontsize=16, fontweight='bold')

        # Original image
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # FFT magnitude spectrum
        im = axes[1].imshow(magnitude_log, cmap='viridis')
        axes[1].set_title('FFT Magnitude Spectrum (log scale)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])

        # Add frequency band annotations
        cy, cx = h // 2, w // 2
        max_radius = min(cy, cx)

        circle_low = plt.Circle((cx, cy), 0.2 * max_radius, fill=False, color='red', linewidth=2, label='Low freq')
        circle_mid = plt.Circle((cx, cy), 0.6 * max_radius, fill=False, color='yellow', linewidth=2, label='Mid freq')
        axes[1].add_patch(circle_low)
        axes[1].add_patch(circle_mid)
        axes[1].legend()

        # Radial frequency profile
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

        radial_profile = np.zeros(max_radius)
        for radius in range(max_radius):
            mask = (r == radius)
            if mask.any():
                radial_profile[radius] = magnitude[mask].mean()

        axes[2].plot(radial_profile)
        axes[2].set_xlabel('Radius (pixels)')
        axes[2].set_ylabel('Mean Magnitude')
        axes[2].set_title('Radial Frequency Profile')
        axes[2].grid(True, alpha=0.3)
        axes[2].axvline(0.2 * max_radius, color='red', linestyle='--', alpha=0.5, label='Low/Mid')
        axes[2].axvline(0.6 * max_radius, color='yellow', linestyle='--', alpha=0.5, label='Mid/High')
        axes[2].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_blob_detection(
        self,
        image: Union[np.ndarray, str, Path],
        save_path: Optional[str] = None
    ):
        """
        Visualize detected blobs overlaid on original image

        Args:
            image: Input image
            save_path: Optional path to save visualization
        """
        rgb, gray = self._load_and_preprocess(image)

        # Invert for blob detection
        gray_inv = 255 - gray

        # Detect blobs
        blobs = blob_log(
            gray_inv,
            min_sigma=self.extractor.blob_min_sigma,
            max_sigma=self.extractor.blob_max_sigma,
            threshold=self.extractor.blob_threshold
        )

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Blob Detection (LoG) - {len(blobs)} blobs detected',
                    fontsize=16, fontweight='bold')

        # Original image
        axes[0].imshow(rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Image with detected blobs
        axes[1].imshow(rgb)
        axes[1].set_title('Detected Blobs (Pores/Vessels)')
        axes[1].axis('off')

        # Draw circles around blobs
        for blob in blobs:
            y, x, sigma = blob
            radius = sigma * np.sqrt(2)
            circle = plt.Circle((x, y), radius, fill=False, color='lime',
                              linewidth=2, alpha=0.8)
            axes[1].add_patch(circle)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_edge_detection(
        self,
        image: Union[np.ndarray, str, Path],
        save_path: Optional[str] = None
    ):
        """
        Visualize edge detection results

        Args:
            image: Input image
            save_path: Optional path to save visualization
        """
        rgb, gray = self._load_and_preprocess(image)

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Adaptive Canny thresholds
        median_mag = np.median(magnitude)
        lower = int(max(0, 0.66 * median_mag))
        upper = int(min(255, 1.33 * median_mag))

        # Canny edge detection
        edges = cv2.Canny(gray, lower, upper)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Edge Detection Analysis', fontsize=16, fontweight='bold')

        # Original image
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Gradient magnitude
        im = axes[0, 1].imshow(magnitude, cmap='hot')
        axes[0, 1].set_title('Gradient Magnitude')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])

        # Canny edges
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title(f'Canny Edges (thresholds: {lower}, {upper})')
        axes[1, 0].axis('off')

        # Edges overlaid on original
        overlay = rgb.copy()
        overlay[edges > 0] = [0, 255, 0]  # Green edges
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Edges Overlaid on Original')
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_lbp_histogram(
        self,
        image: Union[np.ndarray, str, Path],
        save_path: Optional[str] = None
    ):
        """
        Visualize LBP histograms for different radii

        Args:
            image: Input image
            save_path: Optional path to save visualization
        """
        rgb, gray = self._load_and_preprocess(image)

        n_radii = len(self.extractor.lbp_radii)

        fig, axes = plt.subplots(n_radii, 2, figsize=(12, 4 * n_radii))
        fig.suptitle('Local Binary Pattern Analysis', fontsize=16, fontweight='bold')

        for i, radius in enumerate(self.extractor.lbp_radii):
            # Compute LBP
            lbp = local_binary_pattern(
                gray,
                P=self.extractor.lbp_n_points,
                R=radius,
                method='uniform'
            )

            # Compute histogram
            n_bins = self.extractor.lbp_n_points * (self.extractor.lbp_n_points - 1) + 3
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(np.float32) / (hist.sum() + 1e-8)

            # LBP image
            ax_img = axes[i, 0] if n_radii > 1 else axes[0]
            im = ax_img.imshow(lbp, cmap='gray')
            ax_img.set_title(f'LBP (radius={radius})')
            ax_img.axis('off')
            plt.colorbar(im, ax=ax_img, fraction=0.046)

            # LBP histogram
            ax_hist = axes[i, 1] if n_radii > 1 else axes[1]
            ax_hist.bar(range(len(hist)), hist, color='steelblue', alpha=0.7)
            ax_hist.set_xlabel('LBP Pattern')
            ax_hist.set_ylabel('Normalized Frequency')
            ax_hist.set_title(f'LBP Histogram (radius={radius})')
            ax_hist.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_hough_circles(
        self,
        image: Union[np.ndarray, str, Path],
        save_path: Optional[str] = None
    ):
        """
        Visualize Hough circle detection

        Args:
            image: Input image
            save_path: Optional path to save visualization
        """
        rgb, gray = self._load_and_preprocess(image)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Hough circle detection
        radii = np.arange(self.extractor.hough_min_radius, self.extractor.hough_max_radius, 10)

        try:
            hough_res = hough_circle(edges, radii)
            accums, cx, cy, radii_found = hough_circle_peaks(
                hough_res, radii,
                min_xdistance=20,
                min_ydistance=20,
                threshold=0.4 * hough_res.max(),
                num_peaks=10
            )

            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Hough Circle Detection - {len(accums)} circles found',
                        fontsize=16, fontweight='bold')

            # Original
            axes[0].imshow(rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Edges
            axes[1].imshow(edges, cmap='gray')
            axes[1].set_title('Edge Map')
            axes[1].axis('off')

            # Detected circles
            axes[2].imshow(rgb)
            axes[2].set_title('Detected Circles')
            axes[2].axis('off')

            for center_y, center_x, radius in zip(cy, cx, radii_found):
                circle = plt.Circle((center_x, center_y), radius, fill=False,
                                  color='cyan', linewidth=2, alpha=0.8)
                axes[2].add_patch(circle)
                axes[2].plot(center_x, center_y, 'r+', markersize=10)

        except Exception as e:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(rgb)
            ax.set_title(f'Hough Circle Detection Failed: {str(e)}')
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_all_features(
        self,
        image: Union[np.ndarray, str, Path],
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive multi-panel visualization showing all features

        Args:
            image: Input image
            save_path: Optional path to save visualization
        """
        rgb, gray = self._load_and_preprocess(image)

        # Extract features without normalization for visualization
        # Save original normalization setting
        original_normalize = self.extractor.normalize
        self.extractor.normalize = False
        features, feature_names = self.extractor.extract_all_features(rgb)
        # Restore normalization setting
        self.extractor.normalize = original_normalize

        # Create large figure with grid layout
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Comprehensive End-Grain Feature Visualization',
                    fontsize=18, fontweight='bold')

        # 1. Original Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(rgb)
        ax1.set_title('Original Image', fontweight='bold')
        ax1.axis('off')

        # 2. Gradient Orientation Histogram
        ax2 = fig.add_subplot(gs[0, 1])
        grad_hist = self.extractor.extract_gradient_orientation_histogram(rgb)
        angles = np.linspace(0, 360, len(grad_hist), endpoint=False)
        ax2.bar(angles, grad_hist, width=360/len(grad_hist), color='steelblue', alpha=0.7)
        ax2.set_xlabel('Orientation (degrees)')
        ax2.set_ylabel('Normalized Magnitude')
        ax2.set_title('Gradient Orientation', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Edge Detection
        ax3 = fig.add_subplot(gs[0, 2])
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        median_mag = np.median(magnitude)
        edges = cv2.Canny(gray, int(0.66*median_mag), int(1.33*median_mag))
        ax3.imshow(edges, cmap='gray')
        ax3.set_title('Edge Detection', fontweight='bold')
        ax3.axis('off')

        # 4. Blob Detection
        ax4 = fig.add_subplot(gs[0, 3])
        gray_inv = 255 - gray
        blobs = blob_log(gray_inv, min_sigma=1, max_sigma=10, threshold=0.1)
        ax4.imshow(rgb)
        for blob in blobs[:20]:  # Limit to 20 for clarity
            y, x, sigma = blob
            circle = plt.Circle((x, y), sigma*np.sqrt(2), fill=False,
                              color='lime', linewidth=1.5)
            ax4.add_patch(circle)
        ax4.set_title(f'Blob Detection ({len(blobs)} blobs)', fontweight='bold')
        ax4.axis('off')

        # 5. FFT Spectrum
        ax5 = fig.add_subplot(gs[1, 0])
        window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
        fft = fftpack.fft2(gray.astype(np.float32) * window)
        fft_shift = fftpack.fftshift(fft)
        magnitude_log = np.log(np.abs(fft_shift) + 1)
        ax5.imshow(magnitude_log, cmap='viridis')
        ax5.set_title('FFT Spectrum', fontweight='bold')
        ax5.axis('off')

        # 6. LBP Example (radius=2)
        ax6 = fig.add_subplot(gs[1, 1])
        lbp = local_binary_pattern(gray, P=8, R=2, method='uniform')
        im = ax6.imshow(lbp, cmap='gray')
        ax6.set_title('LBP (radius=2)', fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im, ax=ax6, fraction=0.046)

        # 7. Gabor Response Example
        ax7 = fig.add_subplot(gs[1, 2])
        gray_float = gray.astype(np.float32) / 255.0
        gabor_response = cv2.filter2D(gray_float, cv2.CV_32F, self.extractor.gabor_kernels[0])
        im = ax7.imshow(gabor_response, cmap='RdBu_r')
        ax7.set_title('Gabor Response (f=0.05, θ=0°)', fontweight='bold')
        ax7.axis('off')
        plt.colorbar(im, ax=ax7, fraction=0.046)

        # 8. Statistical Distribution
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.hist(gray.ravel(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax8.set_xlabel('Intensity')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Intensity Distribution', fontweight='bold')
        ax8.grid(True, alpha=0.3)

        # 9-12. Feature Statistics Summary
        ax9 = fig.add_subplot(gs[2, :])
        ax9.axis('off')

        # Group features by category
        feature_groups = {
            'Gradient': [f for f in feature_names if f.startswith('grad_')],
            'Gabor': [f for f in feature_names if f.startswith('gabor_')],
            'LBP': [f for f in feature_names if f.startswith('lbp_')],
            'FFT': [f for f in feature_names if f.startswith('fft_')],
            'Wavelet': [f for f in feature_names if f.startswith('wavelet_')],
            'Edge': [f for f in feature_names if f.startswith('edge_')],
            'Blob': [f for f in feature_names if f.startswith('blob_')],
            'Hough': [f for f in feature_names if f.startswith('hough_')],
            'GLCM': [f for f in feature_names if f.startswith('glcm_')],
            'Stats': [f for f in feature_names if f.startswith('stat_')]
        }

        summary_text = "Feature Summary:\n\n"
        for group, feat_list in feature_groups.items():
            count = len(feat_list)
            if count > 0:
                feat_indices = [feature_names.index(f) for f in feat_list]
                feat_values = features[feat_indices]
                mean_val = feat_values.mean()
                std_val = feat_values.std()
                summary_text += f"{group}: {count} features (μ={mean_val:.3f}, σ={std_val:.3f})\n"

        summary_text += f"\nTotal Features: {len(features)}"

        ax9.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def compare_images(
        self,
        image1: Union[np.ndarray, str, Path],
        image2: Union[np.ndarray, str, Path],
        label1: str = "Image 1",
        label2: str = "Image 2",
        save_path: Optional[str] = None
    ):
        """
        Compare feature vectors from two images

        Args:
            image1: First image
            image2: Second image
            label1: Label for first image
            label2: Label for second image
            save_path: Optional path to save visualization
        """
        # Extract features
        features1, names = self.extractor.extract_all_features(image1)
        features2, _ = self.extractor.extract_all_features(image2)

        # Load images
        rgb1, _ = self._load_and_preprocess(image1)
        rgb2, _ = self._load_and_preprocess(image2)

        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Feature Comparison', fontsize=18, fontweight='bold')

        # Show images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(rgb1)
        ax1.set_title(label1, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(rgb2)
        ax2.set_title(label2, fontweight='bold')
        ax2.axis('off')

        # Feature comparison (grouped by category)
        ax3 = fig.add_subplot(gs[1, :])

        # Sample key features for comparison
        key_features = [
            'edge_density', 'edge_mean_strength',
            'blob_count', 'blob_regularity',
            'fft_low_energy', 'fft_mid_energy', 'fft_high_energy',
            'glcm_contrast', 'glcm_homogeneity', 'glcm_energy',
            'stat_mean', 'stat_std', 'stat_entropy'
        ]

        available_features = [f for f in key_features if f in names]
        indices = [names.index(f) for f in available_features]
        values1 = features1[indices]
        values2 = features2[indices]

        x = np.arange(len(available_features))
        width = 0.35

        ax3.bar(x - width/2, values1, width, label=label1, alpha=0.8)
        ax3.bar(x + width/2, values2, width, label=label2, alpha=0.8)
        ax3.set_ylabel('Feature Value')
        ax3.set_title('Key Feature Comparison', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(available_features, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
