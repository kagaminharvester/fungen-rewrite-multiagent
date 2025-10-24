"""
Comprehensive unit tests for core/preprocessing.py

Tests cover:
- Image preprocessing operations (resize, normalize, crop)
- Batch preprocessing
- VR video preprocessing (SBS, TB)
- Data augmentation
- Performance and correctness

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

from core.preprocessing import (
    crop_frame,
    extract_sbs_left,
    extract_sbs_right,
    extract_tb_bottom,
    extract_tb_top,
    normalize_frame,
    preprocess_batch,
    resize_frame,
)

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


# ============================================================================
# Resize Tests
# ============================================================================


def test_resize_frame_basic():
    """Test basic frame resizing."""
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    resized = resize_frame(frame, target_width=640, target_height=480)

    assert resized.shape == (480, 640, 3)
    assert resized.dtype == np.uint8


def test_resize_frame_preserves_aspect():
    """Test resizing preserves aspect ratio when requested."""
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    resized = resize_frame(frame, target_width=640, target_height=480, preserve_aspect=True)

    # Should pad to maintain aspect ratio
    assert resized.shape[0] <= 480
    assert resized.shape[1] <= 640


def test_resize_frame_upscale():
    """Test upscaling frames."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    resized = resize_frame(frame, target_width=1920, target_height=1080)

    assert resized.shape == (1080, 1920, 3)


def test_resize_frame_grayscale():
    """Test resizing grayscale frames."""
    frame = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
    resized = resize_frame(frame, target_width=640, target_height=480)

    assert resized.shape == (480, 640)


# ============================================================================
# Normalization Tests
# ============================================================================


def test_normalize_frame_zero_one():
    """Test normalization to [0, 1] range."""
    frame = np.array([[[0, 128, 255]]], dtype=np.uint8)
    normalized = normalize_frame(frame, mode="zero_one")

    assert normalized.dtype == np.float32
    assert np.min(normalized) >= 0.0
    assert np.max(normalized) <= 1.0
    assert abs(normalized[0, 0, 0] - 0.0) < 0.01
    assert abs(normalized[0, 0, 2] - 1.0) < 0.01


def test_normalize_frame_neg_one_one():
    """Test normalization to [-1, 1] range."""
    frame = np.array([[[0, 128, 255]]], dtype=np.uint8)
    normalized = normalize_frame(frame, mode="neg_one_one")

    assert normalized.dtype == np.float32
    assert np.min(normalized) >= -1.0
    assert np.max(normalized) <= 1.0


def test_normalize_frame_imagenet():
    """Test ImageNet-style normalization."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    normalized = normalize_frame(frame, mode="imagenet")

    assert normalized.dtype == np.float32
    # ImageNet uses mean/std normalization
    assert np.mean(normalized) < 1.0  # Should be centered


def test_normalize_frame_batch():
    """Test batch normalization."""
    frames = np.random.randint(0, 255, (4, 480, 640, 3), dtype=np.uint8)
    normalized = normalize_frame(frames, mode="zero_one")

    assert normalized.shape == (4, 480, 640, 3)
    assert normalized.dtype == np.float32
    assert np.min(normalized) >= 0.0
    assert np.max(normalized) <= 1.0


# ============================================================================
# Crop Tests
# ============================================================================


def test_crop_frame_center():
    """Test center cropping."""
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    cropped = crop_frame(frame, x=460, y=90, width=1000, height=900)

    assert cropped.shape == (900, 1000, 3)


def test_crop_frame_boundaries():
    """Test cropping respects frame boundaries."""
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Crop that would exceed boundaries
    cropped = crop_frame(frame, x=1800, y=1000, width=500, height=500)

    # Should clip to frame boundaries
    assert cropped.shape[0] <= 500
    assert cropped.shape[1] <= 500


def test_crop_frame_roi():
    """Test ROI (region of interest) cropping."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Set ROI to white
    frame[100:200, 100:200] = 255

    cropped = crop_frame(frame, x=100, y=100, width=100, height=100)

    assert cropped.shape == (100, 100, 3)
    assert np.all(cropped == 255)


# ============================================================================
# VR Video Preprocessing Tests
# ============================================================================


def test_extract_sbs_left():
    """Test extracting left eye from side-by-side VR."""
    # Create SBS frame (left=white, right=black)
    frame = np.zeros((1920, 3840, 3), dtype=np.uint8)
    frame[:, :1920] = 255  # Left half white

    left = extract_sbs_left(frame)

    assert left.shape == (1920, 1920, 3)
    assert np.all(left == 255)


def test_extract_sbs_right():
    """Test extracting right eye from side-by-side VR."""
    # Create SBS frame (left=black, right=white)
    frame = np.zeros((1920, 3840, 3), dtype=np.uint8)
    frame[:, 1920:] = 255  # Right half white

    right = extract_sbs_right(frame)

    assert right.shape == (1920, 1920, 3)
    assert np.all(right == 255)


def test_extract_tb_top():
    """Test extracting top eye from top-bottom VR."""
    # Create TB frame (top=white, bottom=black)
    frame = np.zeros((3840, 1920, 3), dtype=np.uint8)
    frame[:1920, :] = 255  # Top half white

    top = extract_tb_top(frame)

    assert top.shape == (1920, 1920, 3)
    assert np.all(top == 255)


def test_extract_tb_bottom():
    """Test extracting bottom eye from top-bottom VR."""
    # Create TB frame (top=black, bottom=white)
    frame = np.zeros((3840, 1920, 3), dtype=np.uint8)
    frame[1920:, :] = 255  # Bottom half white

    bottom = extract_tb_bottom(frame)

    assert bottom.shape == (1920, 1920, 3)
    assert np.all(bottom == 255)


# ============================================================================
# Batch Preprocessing Tests
# ============================================================================


def test_preprocess_batch_basic():
    """Test batch preprocessing pipeline."""
    frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(4)]

    processed = preprocess_batch(
        frames, target_size=(640, 640), normalize=True, normalize_mode="zero_one"
    )

    assert processed.shape == (4, 640, 640, 3)
    assert processed.dtype == np.float32
    assert np.min(processed) >= 0.0
    assert np.max(processed) <= 1.0


def test_preprocess_batch_no_normalize():
    """Test batch preprocessing without normalization."""
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]

    processed = preprocess_batch(frames, target_size=(640, 640), normalize=False)

    assert processed.shape == (4, 640, 640, 3)
    assert processed.dtype == np.uint8


def test_preprocess_batch_with_crop():
    """Test batch preprocessing with cropping."""
    frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(4)]

    processed = preprocess_batch(
        frames,
        target_size=(640, 640),
        crop_roi=(100, 100, 1720, 880),  # x, y, width, height
        normalize=True,
    )

    assert processed.shape == (4, 640, 640, 3)


# ============================================================================
# Performance Tests
# ============================================================================


def test_preprocessing_performance():
    """Test preprocessing performance (should be fast)."""
    import time

    frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(10)]

    start_time = time.time()
    processed = preprocess_batch(frames, target_size=(640, 640), normalize=True)
    elapsed = time.time() - start_time

    # Should process 10 frames in < 1 second on Pi
    assert elapsed < 2.0, f"Preprocessing too slow: {elapsed:.3f}s"


# ============================================================================
# Edge Cases Tests
# ============================================================================


def test_resize_very_small_frame():
    """Test resizing very small frames."""
    frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    resized = resize_frame(frame, target_width=640, target_height=480)

    assert resized.shape == (480, 640, 3)


def test_normalize_all_zeros():
    """Test normalizing frame with all zeros."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    normalized = normalize_frame(frame, mode="zero_one")

    assert np.all(normalized == 0.0)


def test_normalize_all_ones():
    """Test normalizing frame with all max values."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    normalized = normalize_frame(frame, mode="zero_one")

    assert np.allclose(normalized, 1.0)


# ============================================================================
# Main Test Runner
# ============================================================================


def run_all_tests():
    """Run all tests manually if pytest is not available."""
    test_functions = [
        test_resize_frame_basic,
        test_resize_frame_preserves_aspect,
        test_resize_frame_upscale,
        test_resize_frame_grayscale,
        test_normalize_frame_zero_one,
        test_normalize_frame_neg_one_one,
        test_normalize_frame_imagenet,
        test_normalize_frame_batch,
        test_crop_frame_center,
        test_crop_frame_boundaries,
        test_crop_frame_roi,
        test_extract_sbs_left,
        test_extract_sbs_right,
        test_extract_tb_top,
        test_extract_tb_bottom,
        test_preprocess_batch_basic,
        test_preprocess_batch_no_normalize,
        test_preprocess_batch_with_crop,
        test_preprocessing_performance,
        test_resize_very_small_frame,
        test_normalize_all_zeros,
        test_normalize_all_ones,
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 70)
    print("Preprocessing Test Suite")
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
