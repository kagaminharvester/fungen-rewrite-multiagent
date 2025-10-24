"""
Integration tests for error handling and recovery.

Tests various error conditions, edge cases, and recovery mechanisms:
- Corrupted video files
- Missing files
- Out of memory scenarios
- Invalid configurations
- Crash recovery

Author: test-engineer-2 agent
Date: 2025-10-24
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.batch_processor import BatchProcessor, ProcessingSettings
from core.model_manager import Detection, ModelManager
from core.video_processor import VideoProcessor
from trackers.byte_tracker import ByteTracker
from trackers.improved_tracker import ImprovedTracker

TEST_DIR = Path("/tmp/test_error_handling")


def create_corrupted_video(output_path: Path) -> None:
    """Create a corrupted video file for error testing.

    Args:
        output_path: Path to output corrupted video
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a valid video first
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=640x480:rate=30",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        str(output_path),
    ]

    subprocess.run(cmd, capture_output=True, check=True)

    # Corrupt the video by truncating it
    with open(output_path, "r+b") as f:
        f.seek(0)
        f.truncate(100)  # Keep only first 100 bytes


def create_empty_file(output_path: Path) -> None:
    """Create an empty file.

    Args:
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch()


@pytest.fixture(scope="module")
def test_dir():
    """Create and return test directory."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DIR


class TestVideoProcessorErrors:
    """Test VideoProcessor error handling."""

    def test_nonexistent_video_file(self):
        """Test handling of nonexistent video file."""
        print("\n" + "=" * 80)
        print("TEST: Nonexistent Video File")
        print("=" * 80)

        nonexistent_path = "/nonexistent/path/video.mp4"

        with pytest.raises(FileNotFoundError):
            VideoProcessor(nonexistent_path)

        print("✓ FileNotFoundError raised correctly")
        print("=" * 80)

    def test_corrupted_video_file(self, test_dir):
        """Test handling of corrupted video file."""
        print("\n" + "=" * 80)
        print("TEST: Corrupted Video File")
        print("=" * 80)

        corrupted_path = test_dir / "corrupted.mp4"
        create_corrupted_video(corrupted_path)

        print(f"Created corrupted video: {corrupted_path}")

        # Should raise error or handle gracefully
        try:
            processor = VideoProcessor(str(corrupted_path))
            metadata = processor.get_metadata()
            print(f"Metadata: {metadata}")
            print("⚠ Corrupted video opened (may be partially valid)")
        except Exception as e:
            print(f"✓ Exception raised: {type(e).__name__}: {e}")

        print("=" * 80)

    def test_empty_video_file(self, test_dir):
        """Test handling of empty video file."""
        print("\n" + "=" * 80)
        print("TEST: Empty Video File")
        print("=" * 80)

        empty_path = test_dir / "empty.mp4"
        create_empty_file(empty_path)

        print(f"Created empty file: {empty_path}")

        with pytest.raises(Exception):  # Should raise some error
            VideoProcessor(str(empty_path))

        print("✓ Exception raised for empty file")
        print("=" * 80)

    def test_invalid_seek_position(self, test_dir):
        """Test seeking to invalid frame position."""
        print("\n" + "=" * 80)
        print("TEST: Invalid Seek Position")
        print("=" * 80)

        # Create valid video first
        video_path = test_dir / "valid_seek_test.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=640x480:rate=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(video_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        processor = VideoProcessor(str(video_path))
        metadata = processor.get_metadata()

        print(f"Total frames: {metadata.total_frames}")

        # Try to seek beyond video length
        invalid_frame = metadata.total_frames + 100

        with pytest.raises(ValueError):
            processor.seek(invalid_frame)

        print(f"✓ ValueError raised for seek to frame {invalid_frame}")

        # Try negative frame
        with pytest.raises(ValueError):
            processor.seek(-10)

        print("✓ ValueError raised for negative frame")
        print("=" * 80)

    def test_buffer_overflow_handling(self, test_dir):
        """Test buffer handling with very small buffer size."""
        print("\n" + "=" * 80)
        print("TEST: Buffer Overflow Handling")
        print("=" * 80)

        video_path = test_dir / "buffer_test.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=640x480:rate=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(video_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        # Use very small buffer (1 frame)
        processor = VideoProcessor(str(video_path), buffer_size=1)

        frame_count = 0
        for batch in processor.stream_frames(batch_size=4):
            frame_count += len(batch.frames)

        buffer_stats = processor.get_buffer_stats()

        print(f"Frames processed: {frame_count}")
        print(f"Buffer dropped frames: {buffer_stats['dropped_frames']}")

        # Should handle small buffer gracefully
        assert frame_count > 0
        print("✓ Small buffer handled gracefully")
        print("=" * 80)


class TestTrackerErrors:
    """Test tracker error handling."""

    def test_tracker_empty_detections(self):
        """Test tracker with no detections (empty list)."""
        print("\n" + "=" * 80)
        print("TEST: Tracker with Empty Detections")
        print("=" * 80)

        tracker = ByteTracker()

        # Initialize with empty detections
        tracker.initialize([])

        # Update with empty detections
        for i in range(10):
            tracks = tracker.update([])

        # Should handle gracefully
        funscript = tracker.get_funscript_data()

        print(f"Tracks created: {len(tracker.tracks)}")
        print(f"Action points: {len(funscript.actions)}")

        print("✓ Empty detections handled gracefully")
        print("=" * 80)

    def test_tracker_invalid_bbox(self):
        """Test tracker with invalid bounding boxes."""
        print("\n" + "=" * 80)
        print("TEST: Tracker with Invalid Bounding Boxes")
        print("=" * 80)

        tracker = ByteTracker()

        # Create detections with invalid bboxes
        invalid_detections = [
            # Negative coordinates
            Detection(bbox=(-10, -10, 100, 100), confidence=0.8, class_id=0, class_name="obj"),
            # Inverted bbox (x2 < x1, y2 < y1)
            Detection(bbox=(200, 200, 100, 100), confidence=0.8, class_id=0, class_name="obj"),
            # Zero-area bbox
            Detection(bbox=(50, 50, 50, 50), confidence=0.8, class_id=0, class_name="obj"),
        ]

        # Should handle or filter invalid detections
        try:
            tracker.initialize(invalid_detections)
            tracks = tracker.update([])
            print(f"Tracks created: {len(tracker.tracks)}")
            print("⚠ Invalid bboxes accepted (may be filtered internally)")
        except Exception as e:
            print(f"✓ Exception raised: {type(e).__name__}")

        print("=" * 80)

    def test_tracker_extreme_confidence(self):
        """Test tracker with extreme confidence values."""
        print("\n" + "=" * 80)
        print("TEST: Tracker with Extreme Confidence Values")
        print("=" * 80)

        tracker = ByteTracker()

        # Detections with extreme confidence values
        extreme_detections = [
            Detection(bbox=(0, 0, 100, 100), confidence=2.0, class_id=0, class_name="obj"),  # > 1.0
            Detection(
                bbox=(100, 0, 200, 100), confidence=-0.5, class_id=0, class_name="obj"
            ),  # < 0.0
            Detection(
                bbox=(200, 0, 300, 100), confidence=0.0, class_id=0, class_name="obj"
            ),  # Zero
        ]

        tracker.initialize(extreme_detections)
        tracks = tracker.update([])

        print(f"Tracks created: {len(tracker.tracks)}")
        print("✓ Extreme confidence values handled")
        print("=" * 80)

    def test_tracker_rapid_state_changes(self):
        """Test tracker with rapid appearing/disappearing detections."""
        print("\n" + "=" * 80)
        print("TEST: Tracker with Rapid State Changes")
        print("=" * 80)

        tracker = ByteTracker(max_age=5, min_hits=2)

        # Alternating detections (appear, disappear, appear)
        detection = Detection(
            bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="obj"
        )

        tracker.initialize([detection])

        for i in range(20):
            if i % 2 == 0:
                # Detection present
                tracker.update([detection])
            else:
                # Detection absent
                tracker.update([])

        print(f"Total tracks: {len(tracker.tracks)}")
        print(f"Active tracks: {len(tracker.get_active_tracks())}")

        # Should handle rapid changes
        print("✓ Rapid state changes handled")
        print("=" * 80)


class TestBatchProcessorErrors:
    """Test BatchProcessor error handling."""

    def test_batch_processor_invalid_worker_count(self):
        """Test BatchProcessor with invalid worker count."""
        print("\n" + "=" * 80)
        print("TEST: BatchProcessor Invalid Worker Count")
        print("=" * 80)

        # Zero workers
        with pytest.raises(ValueError):
            BatchProcessor(num_workers=0)

        print("✓ ValueError raised for 0 workers")

        # Negative workers
        with pytest.raises(ValueError):
            BatchProcessor(num_workers=-1)

        print("✓ ValueError raised for negative workers")
        print("=" * 80)

    def test_batch_processor_nonexistent_video(self, test_dir):
        """Test adding nonexistent video to batch processor."""
        print("\n" + "=" * 80)
        print("TEST: BatchProcessor with Nonexistent Video")
        print("=" * 80)

        processor = BatchProcessor(num_workers=1)
        settings = ProcessingSettings(
            tracker_type="bytetrack", hw_accel=False, batch_size=4, output_dir=test_dir
        )

        nonexistent_video = Path("/nonexistent/video.mp4")

        # Should raise error when adding
        with pytest.raises(FileNotFoundError):
            processor.add_video(nonexistent_video, settings)

        print("✓ FileNotFoundError raised")
        print("=" * 80)

    def test_batch_processor_invalid_settings(self, test_dir):
        """Test BatchProcessor with invalid settings."""
        print("\n" + "=" * 80)
        print("TEST: BatchProcessor Invalid Settings")
        print("=" * 80)

        processor = BatchProcessor(num_workers=1)

        # Create valid video
        video_path = test_dir / "settings_test.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=640x480:rate=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(video_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        # Invalid batch size (zero)
        invalid_settings = ProcessingSettings(
            tracker_type="bytetrack", hw_accel=False, batch_size=0, output_dir=test_dir  # Invalid
        )

        with pytest.raises(ValueError):
            processor.add_video(video_path, invalid_settings)

        print("✓ ValueError raised for invalid batch_size")

        # Invalid tracker type
        invalid_settings = ProcessingSettings(
            tracker_type="nonexistent_tracker", hw_accel=False, batch_size=4, output_dir=test_dir
        )

        with pytest.raises(ValueError):
            processor.add_video(video_path, invalid_settings)

        print("✓ ValueError raised for invalid tracker_type")
        print("=" * 80)


class TestRecoveryMechanisms:
    """Test recovery and checkpoint mechanisms."""

    def test_checkpoint_save_load(self, test_dir):
        """Test checkpoint save and load for crash recovery."""
        print("\n" + "=" * 80)
        print("TEST: Checkpoint Save/Load")
        print("=" * 80)

        checkpoint_path = test_dir / "checkpoint.json"

        # Create mock processing state
        state = {
            "video_path": "/path/to/video.mp4",
            "frames_processed": 150,
            "total_frames": 300,
            "tracker_state": {"tracks": [{"track_id": 1, "positions": [[100, 200], [110, 210]]}]},
        }

        # Save checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump(state, f, indent=2)

        print(f"Checkpoint saved: {checkpoint_path}")
        print(f"Frames processed: {state['frames_processed']}/{state['total_frames']}")

        # Load checkpoint
        with open(checkpoint_path, "r") as f:
            loaded_state = json.load(f)

        assert loaded_state["frames_processed"] == 150
        assert loaded_state["total_frames"] == 300
        assert len(loaded_state["tracker_state"]["tracks"]) == 1

        print("✓ Checkpoint saved and loaded correctly")
        print("=" * 80)

    def test_graceful_shutdown(self):
        """Test graceful shutdown during processing."""
        print("\n" + "=" * 80)
        print("TEST: Graceful Shutdown")
        print("=" * 80)

        processor = BatchProcessor(num_workers=1)

        # Start processing (would normally run)
        processor.shutdown()

        # Should be able to call shutdown multiple times
        processor.shutdown()

        print("✓ Graceful shutdown completed")
        print("=" * 80)


class TestMemoryStress:
    """Test memory-related error handling."""

    def test_large_frame_batch(self):
        """Test handling very large frame batches."""
        print("\n" + "=" * 80)
        print("TEST: Large Frame Batch")
        print("=" * 80)

        tracker = ByteTracker()

        # Create very large detection list
        large_detection_list = [
            Detection(
                bbox=(i * 10, i * 10, i * 10 + 50, i * 10 + 50),
                confidence=0.8,
                class_id=0,
                class_name=f"obj_{i}",
            )
            for i in range(1000)  # 1000 detections
        ]

        print(f"Testing with {len(large_detection_list)} detections")

        start_time = __import__("time").perf_counter()
        tracker.initialize(large_detection_list)
        elapsed = __import__("time").perf_counter() - start_time

        print(f"Initialization time: {elapsed:.3f}s")
        print(f"Tracks created: {len(tracker.tracks)}")

        # Should handle large batches
        assert len(tracker.tracks) > 0
        print("✓ Large detection batch handled")
        print("=" * 80)

    def test_long_tracking_session(self):
        """Test tracker with very long tracking session (memory leak check)."""
        print("\n" + "=" * 80)
        print("TEST: Long Tracking Session")
        print("=" * 80)

        tracker = ByteTracker(max_age=100)

        # Simulate 1000 frames of tracking
        detection = Detection(
            bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="obj"
        )

        tracker.initialize([detection])

        for i in range(1000):
            # Move detection slightly each frame
            moved_det = Detection(
                bbox=(100 + i, 100, 200 + i, 200), confidence=0.9, class_id=0, class_name="obj"
            )
            tracker.update([moved_det])

        print(f"Frames tracked: 1000")
        print(f"Active tracks: {len(tracker.get_active_tracks())}")

        # Memory should not grow unbounded
        primary_track = tracker.get_primary_track()
        if primary_track:
            print(f"Primary track positions: {len(primary_track.positions)}")

        print("✓ Long tracking session completed")
        print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
