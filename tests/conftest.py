"""
Pytest configuration and shared fixtures for FunGen rewrite test suite.

This module provides:
- GPU mocking for Raspberry Pi testing
- Common test fixtures (videos, models, frames)
- Test utilities and helpers
- Platform-specific test configuration

Author: test-engineer-1 agent
Date: 2025-10-24
Target: 80%+ code coverage with Pi-compatible tests
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator, List, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import platform detection
try:
    from utils.platform_utils import get_device_type, is_cuda_available, is_raspberry_pi

    PLATFORM_UTILS_AVAILABLE = True
except ImportError:
    PLATFORM_UTILS_AVAILABLE = False

    def is_cuda_available():
        return False

    def is_raspberry_pi():
        return True

    def get_device_type():
        return "cpu"


# ============================================================================
# GPU Mocking for Raspberry Pi Testing
# ============================================================================


class MockTensor:
    """Mock PyTorch tensor for CPU-only testing."""

    def __init__(self, data, device="cpu", dtype=None):
        self.data = np.array(data)
        self.device = device
        self.dtype = dtype
        self.shape = self.data.shape

    def cpu(self):
        return MockTensor(self.data, device="cpu", dtype=self.dtype)

    def cuda(self):
        return MockTensor(self.data, device="cuda", dtype=self.dtype)

    def numpy(self):
        return self.data

    def __getitem__(self, idx):
        return MockTensor(self.data[idx], device=self.device, dtype=self.dtype)

    def __len__(self):
        return len(self.data)


class MockCUDA:
    """Mock torch.cuda for CPU-only testing."""

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def get_device_name(device=0):
        return "Mock GPU"

    @staticmethod
    def memory_allocated(device=0):
        return 0

    @staticmethod
    def memory_reserved(device=0):
        return 0

    @staticmethod
    def max_memory_allocated(device=0):
        return 0

    @staticmethod
    def reset_peak_memory_stats(device=0):
        pass


class MockYOLO:
    """Mock YOLO model for testing without actual model files."""

    def __init__(self, model_path="mock_model.pt"):
        self.model_path = model_path
        self.device = "cpu"
        self.names = {0: "object", 1: "person", 2: "car"}

    def to(self, device):
        self.device = device
        return self

    def predict(self, source, **kwargs):
        """Return mock predictions."""
        # Mock result with bounding boxes
        result = Mock()
        result.boxes = Mock()

        # Generate random boxes
        num_boxes = np.random.randint(1, 5)
        result.boxes.xyxy = MockTensor(np.random.rand(num_boxes, 4) * 640)  # Random boxes
        result.boxes.conf = MockTensor(np.random.rand(num_boxes) * 0.5 + 0.5)  # Confidence 0.5-1.0
        result.boxes.cls = MockTensor(np.random.randint(0, 3, size=num_boxes))  # Random classes

        return [result]

    def export(self, format="onnx", **kwargs):
        """Mock model export."""
        return f"mock_model.{format}"


# ============================================================================
# Pytest Fixtures
# ============================================================================

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

    # Create dummy pytest module for when it's not available
    class DummyPytest:
        @staticmethod
        def fixture(func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func

        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    return func

                return decorator

            @staticmethod
            def parametrize(argnames, argvalues):
                def decorator(func):
                    return func

                return decorator

            gpu = staticmethod(lambda func: func)
            slow = staticmethod(lambda func: func)
            pi = staticmethod(lambda func: func)

    pytest = DummyPytest()


@pytest.fixture
def mock_gpu():
    """Mock GPU environment for CPU-only testing."""
    with patch("torch.cuda", MockCUDA()):
        yield MockCUDA()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture
def sample_video_path(temp_dir: Path) -> Path:
    """Create a sample video file for testing."""
    video_path = temp_dir / "test_video.mp4"

    # Create a minimal valid MP4 file using numpy
    # For actual tests, you might want to use cv2.VideoWriter
    # but this avoids dependencies
    video_path.write_bytes(b"fake video data for testing")

    return video_path


@pytest.fixture
def sample_frames() -> np.ndarray:
    """Generate sample video frames for testing."""
    # Create 10 frames of 640x480 RGB
    frames = np.random.randint(0, 255, size=(10, 480, 640, 3), dtype=np.uint8)
    return frames


@pytest.fixture
def sample_detections():
    """Generate sample YOLO detections for testing."""
    from core.model_manager import Detection

    detections = [
        Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="object1"),
        Detection(bbox=(300, 150, 400, 250), confidence=0.85, class_id=0, class_name="object2"),
        Detection(bbox=(500, 300, 600, 400), confidence=0.75, class_id=1, class_name="person"),
    ]

    return detections


@pytest.fixture
def sample_tracks():
    """Generate sample tracks for testing."""
    from trackers.base_tracker import Detection, Track

    tracks = []
    for track_id in range(3):
        track = Track(track_id=track_id)

        # Add detections to track
        for frame_id in range(10):
            det = Detection(
                bbox=(
                    100 + track_id * 150 + frame_id * 5,
                    100,
                    200 + track_id * 150 + frame_id * 5,
                    200,
                ),
                confidence=0.9,
                class_id=0,
                class_name=f"object{track_id}",
                frame_id=frame_id,
                timestamp=float(frame_id) * 0.033,
            )
            track.update(det)

        tracks.append(track)

    return tracks


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing."""
    return MockYOLO()


@pytest.fixture
def sample_config(temp_dir: Path):
    """Generate sample configuration for testing."""
    config = {
        "model_dir": str(temp_dir / "models"),
        "output_dir": str(temp_dir / "output"),
        "device": "cpu",
        "batch_size": 4,
        "max_age": 30,
        "min_hits": 3,
        "iou_threshold": 0.3,
    }
    return config


# ============================================================================
# Test Utilities
# ============================================================================


def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    if PYTEST_AVAILABLE:
        return pytest.mark.skipif(not is_cuda_available(), reason="GPU not available")
    return lambda func: func


def skip_if_not_pi():
    """Skip test if not on Raspberry Pi."""
    if PYTEST_AVAILABLE:
        return pytest.mark.skipif(not is_raspberry_pi(), reason="Not running on Raspberry Pi")
    return lambda func: func


def create_test_video(
    path: Path, duration: float = 1.0, fps: int = 30, width: int = 640, height: int = 480
):
    """Create a test video file using OpenCV."""
    try:
        import cv2
    except ImportError:
        # If cv2 not available, create fake file
        path.write_bytes(b"fake video data")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    num_frames = int(duration * fps)
    for i in range(num_frames):
        # Create frame with moving gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 255 // num_frames  # Red channel
        frame[:, :, 1] = 128  # Green channel
        frame[:, :, 2] = 255 - (i * 255 // num_frames)  # Blue channel

        out.write(frame)

    out.release()


def assert_approx_equal(a: float, b: float, tolerance: float = 1e-6):
    """Assert two floats are approximately equal."""
    assert abs(a - b) < tolerance, f"{a} != {b} (tolerance: {tolerance})"


def assert_bbox_valid(bbox: tuple):
    """Assert bounding box is valid (x1, y1, x2, y2)."""
    assert len(bbox) == 4, f"Invalid bbox length: {len(bbox)}"
    x1, y1, x2, y2 = bbox
    assert x2 > x1, f"Invalid bbox: x2 ({x2}) <= x1 ({x1})"
    assert y2 > y1, f"Invalid bbox: y2 ({y2}) <= y1 ({y1})"
    assert x1 >= 0 and y1 >= 0, f"Negative coordinates: ({x1}, {y1})"


# ============================================================================
# Session-level Setup
# ============================================================================


def pytest_configure(config):
    """Pytest configuration hook."""
    # Register custom markers
    markers = [
        "unit: Unit tests for individual modules",
        "integration: Integration tests for multiple modules",
        "benchmark: Performance benchmark tests",
        "gpu: Tests requiring GPU",
        "slow: Tests that take >1 second",
        "pi: Tests that can run on Raspberry Pi",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on platform."""
    if not is_cuda_available():
        # Skip GPU tests on CPU-only systems
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
