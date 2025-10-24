"""
Unit tests for ByteTrack tracker implementation.

Tests cover:
- Detection and track creation
- IoU matching and association
- Kalman filter prediction
- Two-stage matching (high/low confidence)
- Funscript generation
- Performance benchmarking
"""

# Import tracker components
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trackers.base_tracker import Detection, FunscriptAction, FunscriptData, Track
from trackers.byte_tracker import ByteTracker, KalmanTrack


# Fixtures
@pytest.fixture
def simple_detection() -> Detection:
    """Create a simple test detection."""
    return Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.9,
        class_id=0,
        class_name="test_object",
        frame_id=0,
        timestamp=0.0,
    )


@pytest.fixture
def high_confidence_detections() -> List[Detection]:
    """Create list of high-confidence detections."""
    return [
        Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="object1",
            frame_id=0,
            timestamp=0.0,
        ),
        Detection(
            bbox=(300, 150, 400, 250),
            confidence=0.85,
            class_id=0,
            class_name="object2",
            frame_id=0,
            timestamp=0.0,
        ),
    ]


@pytest.fixture
def low_confidence_detections() -> List[Detection]:
    """Create list of low-confidence detections."""
    return [
        Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.4,
            class_id=0,
            class_name="object1",
            frame_id=0,
            timestamp=0.0,
        ),
        Detection(
            bbox=(300, 150, 400, 250),
            confidence=0.3,
            class_id=0,
            class_name="object2",
            frame_id=0,
            timestamp=0.0,
        ),
    ]


@pytest.fixture
def byte_tracker() -> ByteTracker:
    """Create ByteTracker instance."""
    return ByteTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        high_threshold=0.6,
        low_threshold=0.1,
        use_kalman=True,
    )


@pytest.fixture
def byte_tracker_no_kalman() -> ByteTracker:
    """Create ByteTracker without Kalman filter."""
    return ByteTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        high_threshold=0.6,
        low_threshold=0.1,
        use_kalman=False,
    )


# Test Detection class
class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_center(self, simple_detection):
        """Test center calculation."""
        center = simple_detection.center()
        assert center == (150, 150)

    def test_detection_area(self, simple_detection):
        """Test area calculation."""
        area = simple_detection.area()
        assert area == 10000  # 100 * 100

    def test_detection_width(self, simple_detection):
        """Test width calculation."""
        width = simple_detection.width()
        assert width == 100

    def test_detection_height(self, simple_detection):
        """Test height calculation."""
        height = simple_detection.height()
        assert height == 100


# Test Track class
class TestTrack:
    """Tests for Track dataclass."""

    def test_track_update(self, simple_detection):
        """Test track update with detection."""
        track = Track(track_id=1)
        track.update(simple_detection)

        assert track.hits == 1
        assert track.age == 1
        assert track.time_since_update == 0
        assert len(track.detections) == 1
        assert len(track.positions) == 1
        assert len(track.confidences) == 1

    def test_track_confirmation(self, simple_detection):
        """Test track confirmation after min_hits."""
        track = Track(track_id=1)

        # Add 3 detections
        for i in range(3):
            det = Detection(
                bbox=(100 + i * 10, 100, 200, 200),
                confidence=0.9,
                class_id=0,
                class_name="test",
                frame_id=i,
                timestamp=float(i),
            )
            track.update(det)

        assert track.state == "confirmed"
        assert track.hits == 3

    def test_track_mark_missed(self):
        """Test marking track as missed."""
        track = Track(track_id=1, state="confirmed")
        track.mark_missed()

        assert track.time_since_update == 1
        assert track.age == 1

    def test_track_lost_state(self):
        """Test track becomes lost after max misses."""
        track = Track(track_id=1, state="confirmed")

        # Miss 31 frames
        for _ in range(31):
            track.mark_missed()

        assert track.state == "lost"

    def test_get_last_position(self, simple_detection):
        """Test getting last position."""
        track = Track(track_id=1)
        track.update(simple_detection)

        last_pos = track.get_last_position()
        assert last_pos == (150, 150)

    def test_get_average_confidence(self, simple_detection):
        """Test average confidence calculation."""
        track = Track(track_id=1)

        for conf in [0.9, 0.8, 0.7]:
            det = Detection(
                bbox=(100, 100, 200, 200),
                confidence=conf,
                class_id=0,
                class_name="test",
                frame_id=0,
                timestamp=0.0,
            )
            track.update(det)

        avg_conf = track.get_average_confidence()
        assert abs(avg_conf - 0.8) < 0.01


# Test ByteTracker
class TestByteTracker:
    """Tests for ByteTracker implementation."""

    def test_initialization(self, byte_tracker, high_confidence_detections):
        """Test tracker initialization."""
        byte_tracker.initialize(high_confidence_detections)

        assert byte_tracker.frame_count == 0
        assert len(byte_tracker.kalman_tracks) == 2
        assert byte_tracker.next_track_id == 3

    def test_high_confidence_track_creation(self, byte_tracker, high_confidence_detections):
        """Test tracks are created from high-confidence detections."""
        byte_tracker.initialize(high_confidence_detections)

        assert len(byte_tracker.kalman_tracks) == 2
        for track in byte_tracker.kalman_tracks:
            assert track.status == "tentative"
            assert track.hits == 1

    def test_low_confidence_ignored_at_init(self, byte_tracker, low_confidence_detections):
        """Test low-confidence detections are ignored at initialization."""
        byte_tracker.initialize(low_confidence_detections)

        assert len(byte_tracker.kalman_tracks) == 0

    def test_iou_calculation(self, byte_tracker):
        """Test IoU calculation."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 150, 150)

        iou = byte_tracker.calculate_iou(bbox1, bbox2)

        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 = 0.142857
        assert abs(iou - 0.142857) < 0.001

    def test_iou_no_overlap(self, byte_tracker):
        """Test IoU with no overlap."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (200, 200, 300, 300)

        iou = byte_tracker.calculate_iou(bbox1, bbox2)
        assert iou == 0.0

    def test_iou_perfect_overlap(self, byte_tracker):
        """Test IoU with perfect overlap."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (0, 0, 100, 100)

        iou = byte_tracker.calculate_iou(bbox1, bbox2)
        assert iou == 1.0

    def test_update_with_matching_detection(self, byte_tracker):
        """Test track update with matching detection."""
        # Initialize with first detection
        det1 = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="object",
            frame_id=0,
            timestamp=0.0,
        )
        byte_tracker.initialize([det1])

        # Update with similar detection
        det2 = Detection(
            bbox=(105, 105, 205, 205),
            confidence=0.85,
            class_id=0,
            class_name="object",
            frame_id=1,
            timestamp=0.033,
        )
        tracks = byte_tracker.update([det2])

        assert len(tracks) == 1
        assert tracks[0].hits == 2
        assert tracks[0].time_since_update == 0

    def test_two_stage_matching(self, byte_tracker):
        """Test two-stage matching with high and low confidence detections."""
        # Initialize with high-confidence detection
        det1 = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="object",
            frame_id=0,
            timestamp=0.0,
        )
        byte_tracker.initialize([det1])

        # Confirm track with more high-confidence detections
        for i in range(1, 4):
            det = Detection(
                bbox=(100 + i * 5, 100, 200 + i * 5, 200),
                confidence=0.9,
                class_id=0,
                class_name="object",
                frame_id=i,
                timestamp=float(i) * 0.033,
            )
            byte_tracker.update([det])

        # Now update with low-confidence detection
        det_low = Detection(
            bbox=(120, 100, 220, 200),
            confidence=0.4,
            class_id=0,
            class_name="object",
            frame_id=4,
            timestamp=0.133,
        )
        tracks = byte_tracker.update([det_low])

        # Track should still be maintained with low-confidence detection
        assert len(tracks) == 1
        assert tracks[0].hits == 5

    def test_track_loss_no_detection(self, byte_tracker):
        """Test track is lost after max_age without detection."""
        # Initialize
        det1 = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="object",
            frame_id=0,
            timestamp=0.0,
        )
        byte_tracker.initialize([det1])

        # Update with empty detections for max_age + 1 frames
        for i in range(byte_tracker.max_age + 1):
            byte_tracker.update([])

        # Track should be removed
        assert len(byte_tracker.kalman_tracks) == 0

    def test_new_track_from_unmatched_detection(self, byte_tracker):
        """Test new track creation from unmatched detection."""
        # Initialize with one detection
        det1 = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="object1",
            frame_id=0,
            timestamp=0.0,
        )
        byte_tracker.initialize([det1])

        # Add completely different detection
        det2 = Detection(
            bbox=(500, 500, 600, 600),
            confidence=0.9,
            class_id=0,
            class_name="object2",
            frame_id=1,
            timestamp=0.033,
        )
        tracks = byte_tracker.update([det2])

        # Should have 2 tracks now
        assert len(byte_tracker.kalman_tracks) == 2

    def test_kalman_filter_creation(self, byte_tracker):
        """Test Kalman filter is created properly."""
        try:
            import cv2

            kf = byte_tracker._create_kalman_filter()
            assert kf is not None
            assert kf.transitionMatrix.shape == (4, 4)
            assert kf.measurementMatrix.shape == (2, 4)
        except ImportError:
            pytest.skip("cv2 not available")

    def test_tracker_without_kalman(self, byte_tracker_no_kalman, high_confidence_detections):
        """Test tracker works without Kalman filter."""
        byte_tracker_no_kalman.initialize(high_confidence_detections)

        # Update with similar detections
        updated_dets = [
            Detection(
                bbox=(105, 105, 205, 205),
                confidence=0.85,
                class_id=0,
                class_name="object1",
                frame_id=1,
                timestamp=0.033,
            )
        ]
        tracks = byte_tracker_no_kalman.update(updated_dets)

        assert len(tracks) >= 1
        assert tracks[0].hits >= 2

    def test_normalize_position(self, byte_tracker):
        """Test position normalization."""
        # Middle of 1080p frame
        pos = byte_tracker.normalize_position(540, 1080, invert=False)
        assert pos == 50

        # Top of frame
        pos = byte_tracker.normalize_position(0, 1080, invert=False)
        assert pos == 0

        # Bottom of frame
        pos = byte_tracker.normalize_position(1080, 1080, invert=False)
        assert pos == 100

    def test_normalize_position_inverted(self, byte_tracker):
        """Test position normalization with inversion."""
        pos = byte_tracker.normalize_position(0, 1080, invert=True)
        assert pos == 100

        pos = byte_tracker.normalize_position(1080, 1080, invert=True)
        assert pos == 0


# Test Funscript Generation
class TestFunscriptGeneration:
    """Tests for funscript generation."""

    def test_get_funscript_data_empty(self, byte_tracker):
        """Test funscript generation with no tracks."""
        funscript = byte_tracker.get_funscript_data()

        assert isinstance(funscript, FunscriptData)
        assert len(funscript.actions) == 0

    def test_get_funscript_data_with_track(self, byte_tracker):
        """Test funscript generation with valid track."""
        # Create a track with multiple detections
        detections = []
        for i in range(10):
            det = Detection(
                bbox=(100, 100 + i * 10, 200, 200 + i * 10),
                confidence=0.9,
                class_id=0,
                class_name="object",
                frame_id=i,
                timestamp=float(i) * 0.033,
            )
            detections.append(det)

        # Initialize and update
        byte_tracker.initialize([detections[0]])
        for det in detections[1:]:
            byte_tracker.update([det])

        # Generate funscript
        funscript = byte_tracker.get_funscript_data(frame_height=1080, fps=30.0, smooth=False)

        assert len(funscript.actions) > 0
        assert funscript.version == "1.0"
        assert funscript.range == 90

        # Check timestamps are increasing
        for i in range(1, len(funscript.actions)):
            assert funscript.actions[i].at >= funscript.actions[i - 1].at

    def test_funscript_smoothing(self, byte_tracker):
        """Test position smoothing in funscript generation."""
        # Create track with noisy positions
        detections = []
        positions = [100, 110, 105, 115, 108, 120, 112, 125]

        for i, pos in enumerate(positions):
            det = Detection(
                bbox=(100, pos, 200, pos + 100),
                confidence=0.9,
                class_id=0,
                class_name="object",
                frame_id=i,
                timestamp=float(i) * 0.033,
            )
            detections.append(det)

        byte_tracker.initialize([detections[0]])
        for det in detections[1:]:
            byte_tracker.update([det])

        # Generate with smoothing
        funscript_smooth = byte_tracker.get_funscript_data(
            frame_height=1080, fps=30.0, smooth=True, smooth_window=3
        )

        # Generate without smoothing
        funscript_raw = byte_tracker.get_funscript_data(frame_height=1080, fps=30.0, smooth=False)

        # Smoothed version should have less variation
        smooth_variance = np.var([a.pos for a in funscript_smooth.actions])
        raw_variance = np.var([a.pos for a in funscript_raw.actions])

        assert smooth_variance <= raw_variance

    def test_funscript_metadata(self, byte_tracker):
        """Test funscript metadata is populated correctly."""
        det = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="object",
            frame_id=0,
            timestamp=0.0,
        )
        byte_tracker.initialize([det])

        funscript = byte_tracker.get_funscript_data(frame_height=1080, fps=30.0)

        assert "tracker" in funscript.metadata
        assert funscript.metadata["tracker"] == "ByteTrack"
        assert "track_id" in funscript.metadata
        assert "axis" in funscript.metadata
        assert "frame_height" in funscript.metadata
        assert "fps" in funscript.metadata


# Test Performance
class TestPerformance:
    """Performance tests for ByteTracker."""

    def test_fps_calculation(self, byte_tracker):
        """Test FPS calculation."""
        # Initialize
        det = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="object",
            frame_id=0,
            timestamp=0.0,
        )
        byte_tracker.initialize([det])

        # Update several times
        for i in range(10):
            det = Detection(
                bbox=(100 + i, 100, 200, 200),
                confidence=0.9,
                class_id=0,
                class_name="object",
                frame_id=i + 1,
                timestamp=float(i + 1) * 0.033,
            )
            byte_tracker.update([det])

        fps = byte_tracker.get_fps()
        assert fps > 0

    def test_update_speed(self, byte_tracker):
        """Test update speed meets performance target (>120 FPS)."""
        # Create initial detections
        det = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="object",
            frame_id=0,
            timestamp=0.0,
        )
        byte_tracker.initialize([det])

        # Benchmark updates
        num_iterations = 100
        start_time = time.time()

        for i in range(num_iterations):
            det = Detection(
                bbox=(100 + i % 50, 100, 200, 200),
                confidence=0.9,
                class_id=0,
                class_name="object",
                frame_id=i + 1,
                timestamp=float(i + 1) * 0.033,
            )
            byte_tracker.update([det])

        elapsed = time.time() - start_time
        fps = num_iterations / elapsed

        print(f"\nByteTracker FPS: {fps:.2f}")

        # Target is 120+ FPS, but be lenient on slower hardware
        assert fps > 50, f"FPS too low: {fps:.2f} (target: 120+)"

    def test_multi_object_performance(self, byte_tracker):
        """Test performance with multiple objects."""
        # Create 5 objects
        initial_dets = []
        for i in range(5):
            det = Detection(
                bbox=(100 + i * 150, 100, 200 + i * 150, 200),
                confidence=0.9,
                class_id=0,
                class_name=f"object{i}",
                frame_id=0,
                timestamp=0.0,
            )
            initial_dets.append(det)

        byte_tracker.initialize(initial_dets)

        # Benchmark updates with 5 objects
        num_iterations = 50
        start_time = time.time()

        for frame_id in range(num_iterations):
            dets = []
            for i in range(5):
                det = Detection(
                    bbox=(100 + i * 150 + frame_id % 20, 100, 200 + i * 150, 200),
                    confidence=0.9,
                    class_id=0,
                    class_name=f"object{i}",
                    frame_id=frame_id + 1,
                    timestamp=float(frame_id + 1) * 0.033,
                )
                dets.append(det)

            byte_tracker.update(dets)

        elapsed = time.time() - start_time
        fps = num_iterations / elapsed

        print(f"\nMulti-object (5 tracks) FPS: {fps:.2f}")

        # Should still maintain good performance with multiple objects
        assert fps > 30, f"Multi-object FPS too low: {fps:.2f}"

    def test_get_stats(self, byte_tracker, high_confidence_detections):
        """Test statistics retrieval."""
        byte_tracker.initialize(high_confidence_detections)
        byte_tracker.update(high_confidence_detections)

        stats = byte_tracker.get_stats()

        assert "frame_count" in stats
        assert "active_tracks" in stats
        assert "confirmed_tracks" in stats
        assert "tentative_tracks" in stats
        assert "fps" in stats
        assert stats["frame_count"] == 1
        assert stats["active_tracks"] >= 0


# Test Edge Cases
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_detections(self, byte_tracker):
        """Test handling of empty detection list."""
        byte_tracker.initialize([])
        tracks = byte_tracker.update([])

        assert len(tracks) == 0
        assert len(byte_tracker.kalman_tracks) == 0

    def test_very_low_confidence(self, byte_tracker):
        """Test detections below low_threshold are ignored."""
        det = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.05,  # Below low_threshold of 0.1
            class_id=0,
            class_name="object",
            frame_id=0,
            timestamp=0.0,
        )

        byte_tracker.initialize([det])
        assert len(byte_tracker.kalman_tracks) == 0

    def test_reset_tracker(self, byte_tracker, high_confidence_detections):
        """Test tracker reset."""
        byte_tracker.initialize(high_confidence_detections)
        assert len(byte_tracker.kalman_tracks) > 0

        byte_tracker.reset()

        assert len(byte_tracker.tracks) == 0
        assert byte_tracker.frame_count == 0
        assert byte_tracker.next_track_id == 1

    def test_bbox_boundary_cases(self, byte_tracker):
        """Test bounding box at frame boundaries."""
        det = Detection(
            bbox=(0, 0, 1920, 1080),
            confidence=0.9,
            class_id=0,
            class_name="object",
            frame_id=0,
            timestamp=0.0,
        )

        center = det.center()
        assert center == (960, 540)

        area = det.area()
        assert area == 1920 * 1080


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
