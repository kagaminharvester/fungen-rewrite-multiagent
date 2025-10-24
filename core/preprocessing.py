"""
Video preprocessing utilities.

This module provides utilities for preprocessing video frames before inference,
including cropping, resizing, normalization, and VR-specific transforms.

Author: video-specialist agent
Date: 2025-10-24
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from core.frame_buffer import FrameBatch


@dataclass
class PreprocessConfig:
    """Configuration for frame preprocessing.

    Attributes:
        target_size: Target (width, height) for resizing. None = no resize
        crop_box: (x1, y1, x2, y2) crop region. None = no crop
        normalize: Whether to normalize pixel values to [0, 1]
        mean: Mean values for normalization (RGB order)
        std: Standard deviation for normalization (RGB order)
        maintain_aspect: Maintain aspect ratio when resizing
        interpolation: OpenCV interpolation method
    """

    target_size: Optional[Tuple[int, int]] = None
    crop_box: Optional[Tuple[int, int, int, int]] = None
    normalize: bool = False
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    maintain_aspect: bool = True
    interpolation: int = cv2.INTER_LINEAR


class FramePreprocessor:
    """Preprocessor for video frames.

    This class applies various preprocessing operations to frames or frame batches,
    optimized for both CPU and GPU execution.

    Example:
        >>> config = PreprocessConfig(target_size=(640, 640), normalize=True)
        >>> preprocessor = FramePreprocessor(config)
        >>> processed_frame = preprocessor.process_frame(frame)
    """

    def __init__(self, config: PreprocessConfig):
        """Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame.

        Args:
            frame: Input frame (H, W, C) in RGB format

        Returns:
            Processed frame
        """
        processed = frame

        # Crop
        if self.config.crop_box is not None:
            processed = self._crop(processed, self.config.crop_box)

        # Resize
        if self.config.target_size is not None:
            processed = self._resize(
                processed,
                self.config.target_size,
                self.config.maintain_aspect,
                self.config.interpolation,
            )

        # Normalize
        if self.config.normalize:
            processed = self._normalize(
                processed,
                self.config.mean,
                self.config.std,
            )

        return processed

    def process_batch(self, batch: FrameBatch) -> FrameBatch:
        """Process a batch of frames.

        Args:
            batch: Input frame batch

        Returns:
            Processed frame batch
        """
        processed_frames = [self.process_frame(frame) for frame in batch.frames]

        return FrameBatch(
            frames=processed_frames,
            metadata=batch.metadata,
        )

    @staticmethod
    def _crop(
        frame: np.ndarray,
        crop_box: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Crop frame to specified region.

        Args:
            frame: Input frame (H, W, C)
            crop_box: (x1, y1, x2, y2) crop coordinates

        Returns:
            Cropped frame
        """
        x1, y1, x2, y2 = crop_box
        return frame[y1:y2, x1:x2, :]

    @staticmethod
    def _resize(
        frame: np.ndarray,
        target_size: Tuple[int, int],
        maintain_aspect: bool,
        interpolation: int,
    ) -> np.ndarray:
        """Resize frame to target size.

        Args:
            frame: Input frame (H, W, C)
            target_size: Target (width, height)
            maintain_aspect: Whether to maintain aspect ratio
            interpolation: OpenCV interpolation method

        Returns:
            Resized frame
        """
        target_w, target_h = target_size
        h, w = frame.shape[:2]

        if maintain_aspect:
            # Calculate scale to fit within target size
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize
            resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

            # Pad to target size (center the image)
            if new_w < target_w or new_h < target_h:
                top = (target_h - new_h) // 2
                bottom = target_h - new_h - top
                left = (target_w - new_w) // 2
                right = target_w - new_w - left

                resized = cv2.copyMakeBorder(
                    resized,
                    top,
                    bottom,
                    left,
                    right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )

            return resized
        else:
            # Direct resize (may distort)
            return cv2.resize(frame, target_size, interpolation=interpolation)

    @staticmethod
    def _normalize(
        frame: np.ndarray,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> np.ndarray:
        """Normalize frame pixel values.

        Args:
            frame: Input frame (H, W, C) in uint8
            mean: Mean values for each channel (RGB)
            std: Std dev for each channel (RGB)

        Returns:
            Normalized frame in float32
        """
        # Convert to float [0, 1]
        normalized = frame.astype(np.float32) / 255.0

        # Apply mean and std
        mean_array = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_array = np.array(std, dtype=np.float32).reshape(1, 1, 3)

        normalized = (normalized - mean_array) / std_array

        return normalized


class VRPreprocessor:
    """Preprocessor for VR video formats.

    This class handles VR-specific preprocessing like side-by-side splitting,
    fisheye undistortion, and equirectangular projection.
    """

    @staticmethod
    def split_sbs(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split side-by-side frame into left and right eyes.

        Args:
            frame: SBS frame (H, W, C) where W is double the actual width

        Returns:
            Tuple of (left_eye, right_eye) frames
        """
        h, w = frame.shape[:2]
        mid = w // 2

        left = frame[:, :mid, :]
        right = frame[:, mid:, :]

        return left, right

    @staticmethod
    def split_tb(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split top-bottom frame into left and right eyes.

        Args:
            frame: TB frame (H, W, C) where H is double the actual height

        Returns:
            Tuple of (left_eye, right_eye) frames
        """
        h, w = frame.shape[:2]
        mid = h // 2

        left = frame[:mid, :, :]
        right = frame[mid:, :, :]

        return left, right

    @staticmethod
    def undistort_fisheye(
        frame: np.ndarray,
        fov_degrees: float = 180.0,
    ) -> np.ndarray:
        """Undistort fisheye frame to rectilinear projection.

        Args:
            frame: Fisheye frame (H, W, C)
            fov_degrees: Field of view in degrees

        Returns:
            Undistorted frame

        Note:
            This is a simplified undistortion. Production code should use
            proper camera calibration parameters.
        """
        h, w = frame.shape[:2]

        # Create camera matrix (simplified)
        focal = w / (2 * np.tan(np.radians(fov_degrees / 2)))
        camera_matrix = np.array(
            [
                [focal, 0, w / 2],
                [0, focal, h / 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Fisheye distortion coefficients (simplified)
        dist_coeffs = np.array([-0.2, 0.1, 0, 0], dtype=np.float32)

        # Undistort
        undistorted = cv2.fisheye.undistortImage(
            frame,
            camera_matrix,
            dist_coeffs,
        )

        return undistorted


class FrameAnalyzer:
    """Analyzer for video frame characteristics.

    This class provides utilities for analyzing video frames to determine
    optimal preprocessing parameters, detect scene changes, etc.
    """

    @staticmethod
    def calculate_brightness(frame: np.ndarray) -> float:
        """Calculate average brightness of frame.

        Args:
            frame: Input frame (H, W, C) in RGB

        Returns:
            Average brightness [0, 255]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def calculate_contrast(frame: np.ndarray) -> float:
        """Calculate contrast (standard deviation) of frame.

        Args:
            frame: Input frame (H, W, C) in RGB

        Returns:
            Contrast value [0, 255]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return float(np.std(gray))

    @staticmethod
    def detect_blur(frame: np.ndarray) -> float:
        """Detect blur using Laplacian variance.

        Args:
            frame: Input frame (H, W, C) in RGB

        Returns:
            Blur score (higher = sharper, lower = blurrier)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)

    @staticmethod
    def calculate_histogram(frame: np.ndarray, bins: int = 256) -> np.ndarray:
        """Calculate RGB histogram.

        Args:
            frame: Input frame (H, W, C) in RGB
            bins: Number of histogram bins

        Returns:
            Histogram array (bins, 3) for R, G, B channels
        """
        histograms = []
        for channel in range(3):
            hist = cv2.calcHist([frame], [channel], None, [bins], [0, 256])
            histograms.append(hist.flatten())

        return np.array(histograms).T

    @staticmethod
    def detect_scene_change(
        frame1: np.ndarray,
        frame2: np.ndarray,
        threshold: float = 30.0,
    ) -> bool:
        """Detect scene change between two frames.

        Args:
            frame1: First frame (H, W, C)
            frame2: Second frame (H, W, C)
            threshold: Difference threshold for scene change

        Returns:
            True if scene change detected
        """
        # Calculate absolute difference
        diff = cv2.absdiff(frame1, frame2)
        mean_diff = np.mean(diff)

        return mean_diff > threshold

    @staticmethod
    def calculate_motion(
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> float:
        """Calculate motion magnitude between frames.

        Args:
            frame1: First frame (H, W, C)
            frame2: Second frame (H, W, C)

        Returns:
            Average motion magnitude in pixels
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        # Calculate optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Calculate magnitude
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(np.mean(magnitude))


def create_preprocessing_config_for_model(
    model_name: str,
    input_size: int = 640,
) -> PreprocessConfig:
    """Create preprocessing config optimized for specific model.

    Args:
        model_name: Model name (e.g., "yolo11n", "yolo11s")
        input_size: Model input size (default: 640)

    Returns:
        PreprocessConfig optimized for the model
    """
    # YOLO models typically use square inputs
    if "yolo" in model_name.lower():
        return PreprocessConfig(
            target_size=(input_size, input_size),
            normalize=False,  # YOLO handles normalization internally
            maintain_aspect=True,
            interpolation=cv2.INTER_LINEAR,
        )

    # Generic config
    return PreprocessConfig(
        target_size=(input_size, input_size),
        normalize=True,
        mean=(0.485, 0.456, 0.406),  # ImageNet defaults
        std=(0.229, 0.224, 0.225),
        maintain_aspect=True,
    )
