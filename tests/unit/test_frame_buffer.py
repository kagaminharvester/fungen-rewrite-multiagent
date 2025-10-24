"""
Unit tests for CircularFrameBuffer.

Author: video-specialist agent
Date: 2025-10-24
"""

import numpy as np
import pytest

from core.frame_buffer import CircularFrameBuffer, FrameBatch, FrameMetadata


class TestFrameMetadata:
    """Test FrameMetadata dataclass."""

    def test_frame_metadata_creation(self):
        """Test creating frame metadata."""
        metadata = FrameMetadata(
            frame_number=42,
            timestamp_ms=1400.0,
            width=1920,
            height=1080,
            channels=3,
        )

        assert metadata.frame_number == 42
        assert metadata.timestamp_ms == 1400.0
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.channels == 3


class TestFrameBatch:
    """Test FrameBatch dataclass."""

    def test_frame_batch_creation(self):
        """Test creating a frame batch."""
        frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(4)]
        metadata = [FrameMetadata(i, i * 33.33, 1920, 1080, 3) for i in range(4)]

        batch = FrameBatch(frames=frames, metadata=metadata)

        assert batch.batch_size == 4
        assert len(batch.frames) == 4
        assert len(batch.metadata) == 4

    def test_frame_batch_to_numpy(self):
        """Test converting batch to numpy array."""
        frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(4)]
        metadata = [FrameMetadata(i, i * 33.33, 1920, 1080, 3) for i in range(4)]

        batch = FrameBatch(frames=frames, metadata=metadata)
        array = batch.to_numpy()

        assert array.shape == (4, 1080, 1920, 3)
        assert array.dtype == np.uint8


class TestCircularFrameBuffer:
    """Test CircularFrameBuffer class."""

    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = CircularFrameBuffer(max_frames=60)

        assert buffer.max_frames == 60
        assert buffer.current_size == 0
        assert buffer.is_empty
        assert not buffer.is_full
        assert buffer.dropped_frames == 0

    def test_buffer_invalid_size(self):
        """Test buffer with invalid size."""
        with pytest.raises(ValueError):
            CircularFrameBuffer(max_frames=0)

        with pytest.raises(ValueError):
            CircularFrameBuffer(max_frames=-1)

    def test_add_frame(self):
        """Test adding frames to buffer."""
        buffer = CircularFrameBuffer(max_frames=10)

        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        metadata = FrameMetadata(0, 0.0, 1920, 1080, 3)

        buffer.add_frame(frame, metadata)

        assert buffer.current_size == 1
        assert not buffer.is_empty
        assert not buffer.is_full

    def test_add_frame_invalid_shape(self):
        """Test adding frame with mismatched shape."""
        buffer = CircularFrameBuffer(max_frames=10)

        # Wrong dimensions
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        metadata = FrameMetadata(0, 0.0, 1920, 720, 3)  # Wrong height

        with pytest.raises(ValueError):
            buffer.add_frame(frame, metadata)

        # Wrong channels
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        metadata = FrameMetadata(0, 0.0, 1920, 1080, 1)  # Wrong channels

        with pytest.raises(ValueError):
            buffer.add_frame(frame, metadata)

    def test_buffer_overflow(self):
        """Test buffer overflow behavior."""
        buffer = CircularFrameBuffer(max_frames=5)

        # Add 10 frames (5 over capacity)
        for i in range(10):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            metadata = FrameMetadata(i, i * 33.33, 1920, 1080, 3)
            buffer.add_frame(frame, metadata)

        # Buffer should only hold 5 frames
        assert buffer.current_size == 5
        assert buffer.is_full
        assert buffer.dropped_frames == 5

    def test_get_batch(self):
        """Test getting a batch of frames."""
        buffer = CircularFrameBuffer(max_frames=10)

        # Add 8 frames
        for i in range(8):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            metadata = FrameMetadata(i, i * 33.33, 1920, 1080, 3)
            buffer.add_frame(frame, metadata)

        # Get batch of 4
        batch = buffer.get_batch(4)

        assert batch is not None
        assert batch.batch_size == 4
        assert buffer.current_size == 4  # 4 frames removed

        # Frame numbers should be 0, 1, 2, 3 (FIFO order)
        for i, meta in enumerate(batch.metadata):
            assert meta.frame_number == i

    def test_get_batch_empty_buffer(self):
        """Test getting batch from empty buffer."""
        buffer = CircularFrameBuffer(max_frames=10)

        batch = buffer.get_batch(4)
        assert batch is None

    def test_get_batch_invalid_size(self):
        """Test getting batch with invalid size."""
        buffer = CircularFrameBuffer(max_frames=10)

        with pytest.raises(ValueError):
            buffer.get_batch(0)

        with pytest.raises(ValueError):
            buffer.get_batch(-1)

    def test_get_batch_partial(self):
        """Test getting batch when buffer has fewer frames than requested."""
        buffer = CircularFrameBuffer(max_frames=10)

        # Add 3 frames
        for i in range(3):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            metadata = FrameMetadata(i, i * 33.33, 1920, 1080, 3)
            buffer.add_frame(frame, metadata)

        # Request 5 frames, should get 3
        batch = buffer.get_batch(5)

        assert batch is not None
        assert batch.batch_size == 3
        assert buffer.is_empty

    def test_is_ready(self):
        """Test checking if buffer is ready for batch."""
        buffer = CircularFrameBuffer(max_frames=10)

        # Add 5 frames
        for i in range(5):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            metadata = FrameMetadata(i, i * 33.33, 1920, 1080, 3)
            buffer.add_frame(frame, metadata)

        assert buffer.is_ready(3)  # Has 5, need 3
        assert buffer.is_ready(5)  # Has 5, need 5
        assert not buffer.is_ready(6)  # Has 5, need 6

    def test_peek_batch(self):
        """Test peeking at batch without removing frames."""
        buffer = CircularFrameBuffer(max_frames=10)

        # Add 8 frames
        for i in range(8):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            metadata = FrameMetadata(i, i * 33.33, 1920, 1080, 3)
            buffer.add_frame(frame, metadata)

        # Peek at batch
        batch = buffer.peek_batch(4)

        assert batch is not None
        assert batch.batch_size == 4
        assert buffer.current_size == 8  # No frames removed

        # Peek again should return same frames
        batch2 = buffer.peek_batch(4)
        assert batch2.batch_size == 4
        assert batch.metadata[0].frame_number == batch2.metadata[0].frame_number

    def test_clear(self):
        """Test clearing buffer."""
        buffer = CircularFrameBuffer(max_frames=10)

        # Add frames
        for i in range(5):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            metadata = FrameMetadata(i, i * 33.33, 1920, 1080, 3)
            buffer.add_frame(frame, metadata)

        assert buffer.current_size == 5

        # Clear
        buffer.clear()

        assert buffer.current_size == 0
        assert buffer.is_empty
        assert buffer.dropped_frames == 0

    def test_memory_usage(self):
        """Test memory usage calculation."""
        buffer = CircularFrameBuffer(max_frames=10)

        # Empty buffer
        assert buffer.get_memory_usage_mb() == 0.0

        # Add 1080p RGB frames
        # Each frame: 1920 * 1080 * 3 bytes = 6,220,800 bytes = ~5.93 MB
        for i in range(5):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            metadata = FrameMetadata(i, i * 33.33, 1920, 1080, 3)
            buffer.add_frame(frame, metadata)

        memory_mb = buffer.get_memory_usage_mb()

        # Should be approximately 5 * 5.93 = ~29.65 MB
        expected_mb = 5 * (1920 * 1080 * 3) / (1024 * 1024)
        assert abs(memory_mb - expected_mb) < 1.0  # Within 1 MB

    def test_len(self):
        """Test __len__ method."""
        buffer = CircularFrameBuffer(max_frames=10)

        assert len(buffer) == 0

        for i in range(5):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            metadata = FrameMetadata(i, i * 33.33, 1920, 1080, 3)
            buffer.add_frame(frame, metadata)

        assert len(buffer) == 5

    def test_repr(self):
        """Test __repr__ method."""
        buffer = CircularFrameBuffer(max_frames=10)

        repr_str = repr(buffer)
        assert "CircularFrameBuffer" in repr_str
        assert "0/10" in repr_str

        # Add frames
        for i in range(5):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            metadata = FrameMetadata(i, i * 33.33, 1920, 1080, 3)
            buffer.add_frame(frame, metadata)

        repr_str = repr(buffer)
        assert "5/10" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
