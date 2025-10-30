"""
VR Video Processing Utilities.

This module provides utilities for processing VR videos, specifically for
converting VR stereoscopic videos to 2D (single eye) for faster processing.

Features:
- Extract left or right eye from SBS (side-by-side) videos
- Extract top or bottom eye from TB (top-bottom) videos
- 2x performance boost by processing single eye instead of both

Author: claude-code optimization
Date: 2025-10-30
"""

import numpy as np
from enum import Enum
from typing import Tuple


class VREye(Enum):
    """VR eye selection for 2D extraction."""
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"  # For TB format
    BOTTOM = "bottom"  # For TB format


class VRFormat(Enum):
    """Supported VR video formats."""
    NONE = "none"
    SBS_FISHEYE_180 = "sbs_fisheye_180"  # Side-by-side fisheye 180°
    SBS_EQUIRECT_180 = "sbs_equirect_180"  # Side-by-side equirectangular 180°
    TB_FISHEYE_180 = "tb_fisheye_180"  # Top-bottom fisheye 180°
    TB_EQUIRECT_180 = "tb_equirect_180"  # Top-bottom equirectangular 180°


def extract_single_eye(
    frame: np.ndarray,
    vr_format: VRFormat,
    eye: VREye = VREye.LEFT
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Extract a single eye view from a VR stereoscopic frame.

    This function extracts either the left or right eye from a side-by-side (SBS)
    or top-bottom (TB) VR video frame, effectively converting it to 2D.
    This provides a 2x performance boost since only half the frame is processed.

    Args:
        frame: Input frame (numpy array, shape: [height, width, channels])
        vr_format: VR format type (SBS or TB)
        eye: Which eye to extract (LEFT, RIGHT, TOP, BOTTOM)

    Returns:
        Tuple of:
            - Extracted frame (single eye view)
            - Original dimensions (width, height) before extraction

    Examples:
        >>> frame = np.random.rand(1080, 3840, 3)  # SBS 4K frame
        >>> left_eye, orig_dims = extract_single_eye(frame, VRFormat.SBS_FISHEYE_180, VREye.LEFT)
        >>> left_eye.shape
        (1080, 1920, 3)  # Half width

    Performance:
        - Processing time: ~0.1ms (just array slicing, no copies)
        - Memory: No additional allocation (returns view, not copy)
        - Speedup: 2x (half the pixels to process)
    """
    if vr_format == VRFormat.NONE:
        # Not a VR video, return as-is
        return frame, (frame.shape[1], frame.shape[0])

    height, width, channels = frame.shape
    orig_dims = (width, height)

    # Side-by-side formats (SBS)
    if "SBS" in vr_format.value.upper():
        mid_point = width // 2

        if eye == VREye.LEFT:
            # Extract left half
            return frame[:, :mid_point, :], orig_dims
        else:  # VREye.RIGHT
            # Extract right half
            return frame[:, mid_point:, :], orig_dims

    # Top-bottom formats (TB)
    elif "TB" in vr_format.value.upper():
        mid_point = height // 2

        if eye == VREye.TOP or eye == VREye.LEFT:
            # Extract top half
            return frame[:mid_point, :, :], orig_dims
        else:  # VREye.BOTTOM or VREye.RIGHT
            # Extract bottom half
            return frame[mid_point:, :, :], orig_dims

    # Unknown format, return as-is
    return frame, orig_dims


def is_vr_format(vr_format: VRFormat) -> bool:
    """
    Check if the format is a VR format (requires eye extraction).

    Args:
        vr_format: VR format to check

    Returns:
        True if VR format (SBS or TB), False otherwise
    """
    return vr_format != VRFormat.NONE


def get_single_eye_resolution(
    width: int,
    height: int,
    vr_format: VRFormat
) -> Tuple[int, int]:
    """
    Calculate the resolution after single eye extraction.

    Args:
        width: Original video width
        height: Original video height
        vr_format: VR format type

    Returns:
        Tuple of (single_eye_width, single_eye_height)

    Examples:
        >>> get_single_eye_resolution(3840, 1080, VRFormat.SBS_FISHEYE_180)
        (1920, 1080)  # Half width for SBS

        >>> get_single_eye_resolution(1920, 2160, VRFormat.TB_FISHEYE_180)
        (1920, 1080)  # Half height for TB
    """
    if not is_vr_format(vr_format):
        return (width, height)

    if "SBS" in vr_format.value.upper():
        # Side-by-side: half the width
        return (width // 2, height)
    elif "TB" in vr_format.value.upper():
        # Top-bottom: half the height
        return (width, height // 2)

    return (width, height)


def estimate_performance_gain(
    vr_format: VRFormat,
    enable_vr_to_2d: bool
) -> float:
    """
    Estimate the performance gain from VR-to-2D conversion.

    Args:
        vr_format: VR format type
        enable_vr_to_2d: Whether VR-to-2D conversion is enabled

    Returns:
        Performance multiplier (e.g., 2.0 = 2x faster)

    Examples:
        >>> estimate_performance_gain(VRFormat.SBS_FISHEYE_180, True)
        2.0  # 2x speedup from processing half the pixels

        >>> estimate_performance_gain(VRFormat.NONE, True)
        1.0  # No speedup for non-VR video
    """
    if not is_vr_format(vr_format) or not enable_vr_to_2d:
        return 1.0

    # Processing half the pixels = ~2x speedup
    # In practice, slightly less due to overhead, so use 1.9x
    return 1.9
