"""
Tracking algorithms for FunGen rewrite.

This package contains various tracking implementations:
- base_tracker: Abstract base class defining tracker interface
- byte_tracker: Fast ByteTrack implementation with Kalman filtering
"""

from trackers.base_tracker import BaseTracker, Detection, FunscriptAction, FunscriptData, Track
from trackers.byte_tracker import ByteTracker, KalmanTrack

__all__ = [
    "BaseTracker",
    "Detection",
    "Track",
    "FunscriptAction",
    "FunscriptData",
    "ByteTracker",
    "KalmanTrack",
]
