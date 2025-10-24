"""
Circular frame buffer for efficient video processing.

This module provides a memory-efficient circular buffer implementation for storing
decoded video frames. The buffer prevents OOM issues by limiting the maximum number
of frames in memory while enabling batch processing.

Author: video-specialist agent
Date: 2025-10-24
Target Platform: Raspberry Pi (dev) + RTX 3090 (prod)
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class FrameMetadata:
    """Metadata for a single video frame.

    Attributes:
        frame_number: Sequential frame index in video
        timestamp_ms: Timestamp in milliseconds
        width: Frame width in pixels
        height: Frame height in pixels
        channels: Number of color channels (3 for RGB, 1 for grayscale)
    """

    frame_number: int
    timestamp_ms: float
    width: int
    height: int
    channels: int = 3


@dataclass
class FrameBatch:
    """Batch of video frames for processing.

    Attributes:
        frames: List of numpy arrays (H, W, C) in RGB format
        metadata: List of metadata for each frame
        batch_size: Number of frames in batch
    """

    frames: List[np.ndarray]
    metadata: List[FrameMetadata]

    @property
    def batch_size(self) -> int:
        """Return the number of frames in this batch."""
        return len(self.frames)

    def to_numpy(self) -> np.ndarray:
        """Convert batch to numpy array (N, H, W, C).

        Returns:
            Numpy array with shape (batch_size, height, width, channels)
        """
        return np.stack(self.frames, axis=0)


class CircularFrameBuffer:
    """Memory-efficient circular buffer for video frames.

    This buffer maintains a fixed maximum capacity and automatically evicts
    old frames when full. It supports batch streaming and prevents OOM errors
    by limiting memory usage.

    Example:
        >>> buffer = CircularFrameBuffer(max_frames=60)
        >>> for frame in video_stream:
        ...     buffer.add_frame(frame, metadata)
        ...     if buffer.is_ready(batch_size=8):
        ...         batch = buffer.get_batch(8)
        ...         process_batch(batch)

    Attributes:
        max_frames: Maximum number of frames to keep in memory
        current_size: Current number of frames in buffer
    """

    def __init__(self, max_frames: int = 60):
        """Initialize circular frame buffer.

        Args:
            max_frames: Maximum number of frames to keep in memory (default: 60)
                       This limits memory usage to ~60 frames * 1920*1080*3 bytes
                       = ~360MB for 1080p RGB frames
        """
        if max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}")

        self.max_frames = max_frames
        self._frames: deque = deque(maxlen=max_frames)
        self._metadata: deque = deque(maxlen=max_frames)
        self._dropped_frames = 0

    @property
    def current_size(self) -> int:
        """Return current number of frames in buffer."""
        return len(self._frames)

    @property
    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity."""
        return self.current_size >= self.max_frames

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.current_size == 0

    @property
    def dropped_frames(self) -> int:
        """Return count of frames dropped due to buffer overflow."""
        return self._dropped_frames

    def add_frame(self, frame: np.ndarray, metadata: FrameMetadata) -> None:
        """Add a frame to the buffer.

        If buffer is full, the oldest frame is automatically evicted.

        Args:
            frame: Numpy array (H, W, C) in RGB format
            metadata: Frame metadata

        Raises:
            ValueError: If frame shape doesn't match metadata
        """
        # Validate frame shape
        if frame.shape[0] != metadata.height or frame.shape[1] != metadata.width:
            raise ValueError(
                f"Frame shape {frame.shape[:2]} doesn't match metadata "
                f"({metadata.height}, {metadata.width})"
            )

        if frame.shape[2] != metadata.channels:
            raise ValueError(
                f"Frame channels {frame.shape[2]} doesn't match metadata " f"({metadata.channels})"
            )

        # Track dropped frames
        if self.is_full:
            self._dropped_frames += 1

        # Add to buffer (deque handles eviction automatically)
        self._frames.append(frame)
        self._metadata.append(metadata)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough frames for a batch.

        Args:
            batch_size: Desired batch size

        Returns:
            True if buffer contains at least batch_size frames
        """
        return self.current_size >= batch_size

    def get_batch(self, batch_size: int) -> Optional[FrameBatch]:
        """Extract a batch of frames from the buffer.

        Frames are removed from the buffer in FIFO order.

        Args:
            batch_size: Number of frames to extract

        Returns:
            FrameBatch with up to batch_size frames, or None if buffer is empty

        Raises:
            ValueError: If batch_size < 1
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if self.is_empty:
            return None

        # Extract up to batch_size frames
        actual_batch_size = min(batch_size, self.current_size)
        frames = []
        metadata = []

        for _ in range(actual_batch_size):
            frames.append(self._frames.popleft())
            metadata.append(self._metadata.popleft())

        return FrameBatch(frames=frames, metadata=metadata)

    def peek_batch(self, batch_size: int) -> Optional[FrameBatch]:
        """Peek at the next batch without removing frames.

        Args:
            batch_size: Number of frames to peek

        Returns:
            FrameBatch with up to batch_size frames, or None if buffer is empty
        """
        if self.is_empty:
            return None

        actual_batch_size = min(batch_size, self.current_size)
        frames = list(self._frames)[:actual_batch_size]
        metadata = list(self._metadata)[:actual_batch_size]

        return FrameBatch(frames=frames, metadata=metadata)

    def clear(self) -> None:
        """Clear all frames from buffer."""
        self._frames.clear()
        self._metadata.clear()
        self._dropped_frames = 0

    def get_memory_usage_mb(self) -> float:
        """Estimate current memory usage in megabytes.

        Returns:
            Approximate memory usage in MB
        """
        if self.is_empty:
            return 0.0

        # Calculate memory for frames
        total_bytes = sum(frame.nbytes for frame in self._frames)
        return total_bytes / (1024 * 1024)

    def __len__(self) -> int:
        """Return number of frames in buffer."""
        return self.current_size

    def __repr__(self) -> str:
        """Return string representation of buffer state."""
        return (
            f"CircularFrameBuffer(size={self.current_size}/{self.max_frames}, "
            f"memory={self.get_memory_usage_mb():.1f}MB, "
            f"dropped={self.dropped_frames})"
        )
