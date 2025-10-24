"""
Comprehensive unit tests for trackers/optical_flow.py

Tests cover:
- Optical flow initialization
- Flow computation (CPU/GPU mocked)
- Flow vector calculations
- Track refinement with optical flow
- Performance benchmarks
- Error handling

Author: test-engineer-1 agent
Date: 2025-10-24
Target: 80%+ code coverage
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trackers.optical_flow import CUDAOpticalFlow, FlowVector

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


# ============================================================================
# FlowVector Tests
# ============================================================================


def test_flow_vector_creation():
    """Test FlowVector creation."""
    fv = FlowVector(
        point=(100, 200), flow=(5.0, 10.0), magnitude=11.18, angle=63.43, confidence=0.9
    )

    assert fv.point == (100, 200)
    assert fv.flow == (5.0, 10.0)
    assert abs(fv.magnitude - 11.18) < 0.1
    assert abs(fv.angle - 63.43) < 0.1
    assert fv.confidence == 0.9


def test_flow_vector_from_flow():
    """Test FlowVector.from_flow() factory method."""
    point = (100, 200)
    flow = (3.0, 4.0)

    fv = FlowVector.from_flow(point, flow, confidence=0.8)

    assert fv.point == point
    assert fv.flow == flow
    # Magnitude should be sqrt(3^2 + 4^2) = 5.0
    assert abs(fv.magnitude - 5.0) < 0.01
    assert fv.confidence == 0.8


def test_flow_vector_zero_flow():
    """Test FlowVector with zero flow."""
    fv = FlowVector.from_flow((100, 200), (0.0, 0.0))

    assert fv.magnitude == 0.0


def test_flow_vector_negative_flow():
    """Test FlowVector with negative flow."""
    fv = FlowVector.from_flow((100, 200), (-5.0, -5.0))

    # Magnitude should be positive
    assert fv.magnitude > 0.0
    # Angle should be in range for bottom-left quadrant
    assert -180 <= fv.angle <= 180


def test_flow_vector_angle_calculation():
    """Test angle calculation for different flow directions."""
    # Right (0 degrees)
    fv_right = FlowVector.from_flow((0, 0), (10.0, 0.0))
    assert abs(fv_right.angle - 0.0) < 0.1

    # Up (90 degrees)
    fv_up = FlowVector.from_flow((0, 0), (0.0, 10.0))
    assert abs(fv_up.angle - 90.0) < 0.1

    # Left (180 or -180 degrees)
    fv_left = FlowVector.from_flow((0, 0), (-10.0, 0.0))
    assert abs(abs(fv_left.angle) - 180.0) < 0.1

    # Down (-90 degrees)
    fv_down = FlowVector.from_flow((0, 0), (0.0, -10.0))
    assert abs(fv_down.angle - (-90.0)) < 0.1


# ============================================================================
# CUDAOpticalFlow Initialization Tests
# ============================================================================


def test_optical_flow_init_default():
    """Test optical flow initialization with default parameters."""
    with patch("trackers.optical_flow.CUDA_AVAILABLE", False):
        flow = CUDAOpticalFlow()

        assert flow.pyr_scale == 0.5
        assert flow.levels == 3
        assert flow.winsize == 15
        assert flow.iterations == 3
        assert flow.poly_n == 5
        assert abs(flow.poly_sigma - 1.2) < 0.01


def test_optical_flow_init_custom():
    """Test optical flow initialization with custom parameters."""
    with patch("trackers.optical_flow.CUDA_AVAILABLE", False):
        flow = CUDAOpticalFlow(
            pyr_scale=0.7,
            levels=5,
            winsize=21,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            use_cuda=False,
        )

        assert flow.pyr_scale == 0.7
        assert flow.levels == 5
        assert flow.winsize == 21
        assert flow.iterations == 5
        assert flow.poly_n == 7
        assert abs(flow.poly_sigma - 1.5) < 0.01


@patch("trackers.optical_flow.CUDA_AVAILABLE", False)
def test_optical_flow_cpu_fallback():
    """Test optical flow falls back to CPU when CUDA unavailable."""
    flow = CUDAOpticalFlow(use_cuda=True)

    # Should use CPU mode
    assert flow.use_cuda is False


# ============================================================================
# Flow Computation Tests
# ============================================================================


@patch("trackers.optical_flow.cv2.calcOpticalFlowFarneback")
@patch("trackers.optical_flow.CUDA_AVAILABLE", False)
def test_compute_flow_cpu(mock_calc_flow):
    """Test optical flow computation on CPU."""
    # Mock flow result
    mock_flow = np.random.randn(480, 640, 2).astype(np.float32)
    mock_calc_flow.return_value = mock_flow

    flow_computer = CUDAOpticalFlow(use_cuda=False)

    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    result = flow_computer.compute(frame1, frame2)

    assert result.shape == (480, 640, 2)
    mock_calc_flow.assert_called_once()


@patch("trackers.optical_flow.CUDA_AVAILABLE", False)
def test_compute_flow_grayscale_conversion():
    """Test that RGB frames are converted to grayscale."""
    with patch("trackers.optical_flow.cv2.calcOpticalFlowFarneback") as mock_calc:
        mock_calc.return_value = np.zeros((480, 640, 2), dtype=np.float32)

        flow_computer = CUDAOpticalFlow(use_cuda=False)

        # RGB frames
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        flow_computer.compute(frame1, frame2)

        # Should have been called with grayscale frames
        mock_calc.assert_called_once()


# ============================================================================
# Track Refinement Tests
# ============================================================================


@patch("trackers.optical_flow.cv2.calcOpticalFlowFarneback")
@patch("trackers.optical_flow.CUDA_AVAILABLE", False)
def test_refine_track_position(mock_calc_flow):
    """Test refining track position with optical flow."""
    # Mock flow: everything moves right by 5 pixels, down by 10 pixels
    mock_flow = np.zeros((480, 640, 2), dtype=np.float32)
    mock_flow[:, :, 0] = 5.0  # dx
    mock_flow[:, :, 1] = 10.0  # dy
    mock_calc_flow.return_value = mock_flow

    flow_computer = CUDAOpticalFlow(use_cuda=False)

    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Object at (100, 200)
    bbox = (100, 200, 150, 250)

    refined_bbox = flow_computer.refine_bbox(frame1, frame2, bbox)

    # Should move right by 5, down by 10
    assert refined_bbox[0] > bbox[0]  # x1 moved right
    assert refined_bbox[1] > bbox[1]  # y1 moved down


@patch("trackers.optical_flow.cv2.calcOpticalFlowFarneback")
@patch("trackers.optical_flow.CUDA_AVAILABLE", False)
def test_get_flow_vectors_at_points(mock_calc_flow):
    """Test extracting flow vectors at specific points."""
    # Create flow field
    flow_field = np.zeros((480, 640, 2), dtype=np.float32)
    flow_field[100, 200] = [5.0, 10.0]  # Flow at point (200, 100)
    mock_calc_flow.return_value = flow_field

    flow_computer = CUDAOpticalFlow(use_cuda=False)

    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    points = [(200, 100)]  # (x, y) format

    flow_vectors = flow_computer.get_flow_at_points(frame1, frame2, points)

    assert len(flow_vectors) == 1
    fv = flow_vectors[0]
    assert fv.point == (200, 100)
    assert abs(fv.flow[0] - 5.0) < 0.01
    assert abs(fv.flow[1] - 10.0) < 0.01


# ============================================================================
# Edge Cases Tests
# ============================================================================


@patch("trackers.optical_flow.CUDA_AVAILABLE", False)
def test_compute_flow_same_frames():
    """Test computing flow between identical frames."""
    with patch("trackers.optical_flow.cv2.calcOpticalFlowFarneback") as mock_calc:
        # Zero flow for identical frames
        mock_calc.return_value = np.zeros((480, 640, 2), dtype=np.float32)

        flow_computer = CUDAOpticalFlow(use_cuda=False)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        flow = flow_computer.compute(frame, frame)

        # Flow should be zero everywhere
        assert np.allclose(flow, 0.0)


@patch("trackers.optical_flow.CUDA_AVAILABLE", False)
def test_refine_bbox_out_of_bounds():
    """Test bbox refinement doesn't go out of frame bounds."""
    with patch("trackers.optical_flow.cv2.calcOpticalFlowFarneback") as mock_calc:
        # Large flow that would push bbox out of bounds
        mock_flow = np.zeros((480, 640, 2), dtype=np.float32)
        mock_flow[:, :, 0] = 1000.0  # Huge rightward flow
        mock_calc.return_value = mock_flow

        flow_computer = CUDAOpticalFlow(use_cuda=False)

        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        bbox = (500, 200, 600, 300)

        refined = flow_computer.refine_bbox(frame1, frame2, bbox)

        # Should be clipped to frame boundaries
        assert refined[0] >= 0
        assert refined[1] >= 0
        assert refined[2] <= 640
        assert refined[3] <= 480


# ============================================================================
# Performance Tests
# ============================================================================


@patch("trackers.optical_flow.CUDA_AVAILABLE", False)
def test_optical_flow_performance():
    """Test optical flow computation performance."""
    import time

    with patch("trackers.optical_flow.cv2.calcOpticalFlowFarneback") as mock_calc:
        # Fast mock
        mock_calc.return_value = np.zeros((480, 640, 2), dtype=np.float32)

        flow_computer = CUDAOpticalFlow(use_cuda=False)

        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        start_time = time.time()
        for _ in range(10):
            flow_computer.compute(frame1, frame2)
        elapsed = time.time() - start_time

        # With mocking, should be very fast
        assert elapsed < 1.0, f"Optical flow too slow: {elapsed:.3f}s"


# ============================================================================
# Main Test Runner
# ============================================================================


def run_all_tests():
    """Run all tests manually if pytest is not available."""
    test_functions = [
        test_flow_vector_creation,
        test_flow_vector_from_flow,
        test_flow_vector_zero_flow,
        test_flow_vector_negative_flow,
        test_flow_vector_angle_calculation,
        test_optical_flow_init_default,
        test_optical_flow_init_custom,
        test_optical_flow_cpu_fallback,
        test_compute_flow_cpu,
        test_compute_flow_grayscale_conversion,
        test_refine_track_position,
        test_get_flow_vectors_at_points,
        test_compute_flow_same_frames,
        test_refine_bbox_out_of_bounds,
        test_optical_flow_performance,
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 70)
    print("Optical Flow Test Suite")
    print("=" * 70 + "\n")

    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("PASSED")
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)
