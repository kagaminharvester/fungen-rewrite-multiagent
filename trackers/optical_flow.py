"""
CUDA-accelerated optical flow for motion tracking.

This module provides high-performance optical flow computation using GPU acceleration.
It supports Farneback optical flow with CUDA and falls back to CPU when GPU is unavailable.

Author: tracker-dev-2 agent
Date: 2025-10-24
Target Platform: Raspberry Pi (dev) + RTX 3090 (prod)
Performance Target: 100+ FPS (1080p on RTX 3090), 5-10x faster than CPU
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    # Check for CUDA-enabled OpenCV
    if CV2_AVAILABLE and hasattr(cv2, "cuda"):
        CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
    else:
        CUDA_AVAILABLE = False
except Exception:
    CUDA_AVAILABLE = False


@dataclass
class FlowVector:
    """Optical flow vector for a tracked point.

    Attributes:
        point: Original (x, y) point
        flow: Flow vector (dx, dy)
        magnitude: Flow magnitude (speed)
        angle: Flow angle in degrees
        confidence: Flow confidence score (0-1)
    """

    point: Tuple[int, int]
    flow: Tuple[float, float]
    magnitude: float
    angle: float
    confidence: float = 1.0

    @staticmethod
    def from_flow(
        point: Tuple[int, int], flow: Tuple[float, float], confidence: float = 1.0
    ) -> "FlowVector":
        """Create FlowVector from point and flow.

        Args:
            point: Original (x, y) point
            flow: Flow vector (dx, dy)
            confidence: Flow confidence score

        Returns:
            FlowVector object
        """
        dx, dy = flow
        magnitude = np.sqrt(dx**2 + dy**2)
        angle = np.degrees(np.arctan2(dy, dx))
        return FlowVector(point, flow, magnitude, angle, confidence)


class CUDAOpticalFlow:
    """CUDA-accelerated optical flow using Farneback algorithm.

    This class provides high-performance optical flow computation on GPU.
    It automatically falls back to CPU if CUDA is not available.
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        use_cuda: bool = True,
    ):
        """Initialize CUDA optical flow.

        Args:
            pyr_scale: Pyramid scale (<1 for multiple levels)
            levels: Number of pyramid levels
            winsize: Window size for averaging
            iterations: Number of iterations at each pyramid level
            poly_n: Size of pixel neighborhood for polynomial expansion
            poly_sigma: Standard deviation for Gaussian weighting
            use_cuda: Whether to use CUDA acceleration if available
        """
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.use_cuda = use_cuda and CUDA_AVAILABLE

        # Previous frame (for flow computation)
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_frame_gpu: Optional[any] = None

        # CUDA flow object (initialized on first use)
        self.cuda_flow: Optional[any] = None

        # Performance metrics
        self.computation_times: List[float] = []

    def _init_cuda_flow(self):
        """Initialize CUDA optical flow object."""
        if not self.use_cuda or self.cuda_flow is not None:
            return

        self.cuda_flow = cv2.cuda.FarnebackOpticalFlow_create(
            numLevels=self.levels,
            pyrScale=self.pyr_scale,
            fastPyramids=False,
            winSize=self.winsize,
            numIters=self.iterations,
            polyN=self.poly_n,
            polySigma=self.poly_sigma,
            flags=0,
        )

    def compute_flow(
        self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[np.ndarray]:
        """Compute optical flow between previous and current frame.

        Args:
            frame: Current frame (grayscale or BGR)
            roi: Optional region of interest (x1, y1, x2, y2)

        Returns:
            Flow field (H x W x 2) with (dx, dy) at each pixel, or None on first frame
        """
        import time

        start_time = time.time()

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply ROI if specified
        if roi is not None:
            x1, y1, x2, y2 = roi
            gray = gray[y1:y2, x1:x2]

        # Return None on first frame
        if self.prev_frame is None:
            self.prev_frame = gray.copy()
            if self.use_cuda:
                self.prev_frame_gpu = cv2.cuda_GpuMat()
                self.prev_frame_gpu.upload(gray)
            return None

        # Compute flow
        if self.use_cuda:
            flow = self._compute_flow_cuda(gray)
        else:
            flow = self._compute_flow_cpu(self.prev_frame, gray)

        # Update previous frame
        self.prev_frame = gray.copy()
        if self.use_cuda and self.prev_frame_gpu is not None:
            self.prev_frame_gpu.upload(gray)

        # Track computation time
        elapsed = time.time() - start_time
        self.computation_times.append(elapsed)
        if len(self.computation_times) > 30:
            self.computation_times.pop(0)

        return flow

    def _compute_flow_cuda(self, current_frame: np.ndarray) -> np.ndarray:
        """Compute optical flow using CUDA.

        Args:
            current_frame: Current grayscale frame

        Returns:
            Flow field (H x W x 2)
        """
        self._init_cuda_flow()

        # Upload current frame to GPU
        curr_gpu = cv2.cuda_GpuMat()
        curr_gpu.upload(current_frame)

        # Compute flow on GPU
        flow_gpu = self.cuda_flow.calc(self.prev_frame_gpu, curr_gpu, None)

        # Download result
        flow = flow_gpu.download()

        return flow

    def _compute_flow_cpu(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """Compute optical flow using CPU (fallback).

        Args:
            prev_frame: Previous grayscale frame
            curr_frame: Current grayscale frame

        Returns:
            Flow field (H x W x 2)
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame,
            curr_frame,
            None,
            self.pyr_scale,
            self.levels,
            self.winsize,
            self.iterations,
            self.poly_n,
            self.poly_sigma,
            0,
        )

        return flow

    def get_flow_at_points(
        self, flow: np.ndarray, points: List[Tuple[int, int]]
    ) -> List[FlowVector]:
        """Extract flow vectors at specific points.

        Args:
            flow: Flow field (H x W x 2)
            points: List of (x, y) points

        Returns:
            List of FlowVector objects
        """
        if flow is None:
            return []

        flow_vectors = []
        h, w = flow.shape[:2]

        for x, y in points:
            # Boundary check
            if 0 <= y < h and 0 <= x < w:
                dx, dy = flow[y, x]
                flow_vectors.append(FlowVector.from_flow((x, y), (float(dx), float(dy))))

        return flow_vectors

    def get_average_flow_in_bbox(
        self, flow: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[Tuple[float, float]]:
        """Compute average flow within a bounding box.

        Args:
            flow: Flow field (H x W x 2)
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Average (dx, dy) flow or None if invalid
        """
        if flow is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = flow.shape[:2]

        # Clamp bbox to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Extract flow in bbox
        flow_roi = flow[y1:y2, x1:x2]

        # Compute average
        avg_dx = float(np.mean(flow_roi[:, :, 0]))
        avg_dy = float(np.mean(flow_roi[:, :, 1]))

        return (avg_dx, avg_dy)

    def visualize_flow(self, flow: np.ndarray, step: int = 16, scale: float = 3.0) -> np.ndarray:
        """Create visualization of optical flow field.

        Args:
            flow: Flow field (H x W x 2)
            step: Sampling step for arrows
            scale: Arrow scale multiplier

        Returns:
            Flow visualization image (BGR)
        """
        if flow is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = flow.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)

        # Sample flow field
        y, x = np.mgrid[step // 2 : h : step, step // 2 : w : step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        # Draw arrows
        lines = np.vstack([x, y, x + fx * scale, y + fy * scale]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)

        for (x1, y1), (x2, y2) in lines:
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

        return vis

    def compute_flow_magnitude(self, flow: np.ndarray) -> np.ndarray:
        """Compute magnitude of flow field.

        Args:
            flow: Flow field (H x W x 2)

        Returns:
            Magnitude field (H x W)
        """
        if flow is None:
            return np.zeros((480, 640), dtype=np.float32)

        magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        return magnitude

    def detect_motion_regions(self, flow: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """Detect regions with significant motion.

        Args:
            flow: Flow field (H x W x 2)
            threshold: Magnitude threshold for motion detection

        Returns:
            Binary mask (H x W) with 1 for motion regions
        """
        if flow is None:
            return np.zeros((480, 640), dtype=np.uint8)

        magnitude = self.compute_flow_magnitude(flow)
        motion_mask = (magnitude > threshold).astype(np.uint8) * 255

        return motion_mask

    def get_fps(self) -> float:
        """Calculate optical flow computation FPS.

        Returns:
            Average FPS over recent frames
        """
        if not self.computation_times:
            return 0.0

        avg_time = np.mean(self.computation_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def reset(self):
        """Reset optical flow state."""
        self.prev_frame = None
        self.prev_frame_gpu = None
        self.computation_times = []


class SparseOpticalFlow:
    """Sparse optical flow using Lucas-Kanade for tracking specific points.

    This is more efficient than dense flow when only tracking a few points.
    """

    def __init__(
        self,
        win_size: Tuple[int, int] = (21, 21),
        max_level: int = 3,
        use_cuda: bool = True,
    ):
        """Initialize sparse optical flow.

        Args:
            win_size: Window size for Lucas-Kanade
            max_level: Maximum pyramid level
            use_cuda: Whether to use CUDA acceleration if available
        """
        self.win_size = win_size
        self.max_level = max_level
        self.use_cuda = use_cuda and CUDA_AVAILABLE

        self.prev_frame: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None

        # LK parameters
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def track_points(
        self, frame: np.ndarray, points: List[Tuple[float, float]]
    ) -> List[Optional[Tuple[float, float]]]:
        """Track points using Lucas-Kanade optical flow.

        Args:
            frame: Current frame (grayscale or BGR)
            points: List of (x, y) points to track

        Returns:
            List of tracked (x, y) points (None for lost tracks)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Initialize on first frame
        if self.prev_frame is None:
            self.prev_frame = gray.copy()
            self.prev_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            return points

        # Convert points to numpy array
        prev_pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

        # Calculate optical flow
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, prev_pts, None, **self.lk_params
        )

        # Update state
        self.prev_frame = gray.copy()
        self.prev_points = next_pts

        # Return tracked points (None for lost tracks)
        tracked = []
        for i, (pt, st) in enumerate(zip(next_pts, status)):
            if st[0] == 1:
                tracked.append((float(pt[0][0]), float(pt[0][1])))
            else:
                tracked.append(None)

        return tracked

    def reset(self):
        """Reset sparse optical flow state."""
        self.prev_frame = None
        self.prev_points = None
