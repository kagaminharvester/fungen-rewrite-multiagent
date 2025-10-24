"""
Unit tests for improved tracker components.

Tests cover:
- Kalman filter prediction and update
- Optical flow computation
- Improved tracker integration
- Performance metrics

Author: tracker-dev-2 agent
Date: 2025-10-24
"""

# Import tracker components
import sys
from typing import List

import numpy as np
import pytest

sys.path.insert(0, "/home/pi/elo_elo_320")

from trackers.base_tracker import Detection, Track
from trackers.improved_tracker import ImprovedTrack, ImprovedTracker
from trackers.kalman_filter import AdvancedKalmanFilter, KalmanFilterCV2, KalmanState
from trackers.optical_flow import CUDAOpticalFlow, FlowVector, SparseOpticalFlow


class TestAdvancedKalmanFilter:
    """Test suite for Advanced Kalman Filter."""

    def test_initialization(self):
        """Test Kalman filter initialization."""
        kf = AdvancedKalmanFilter(dt=1.0, process_noise=0.03, measurement_noise=1.0)

        # Test state initialization
        state = kf.initialize(position=(100.0, 200.0), track_id=1)

        assert state.track_id == 1
        assert state.x.shape == (6, 1)
        assert state.P.shape == (6, 6)
        assert state.x[0, 0] == 100.0  # x position
        assert state.x[1, 0] == 200.0  # y position
        assert state.x[2, 0] == 0.0  # vx velocity
        assert state.x[3, 0] == 0.0  # vy velocity

    def test_predict(self):
        """Test Kalman filter prediction step."""
        kf = AdvancedKalmanFilter(dt=1.0)
        state = kf.initialize((100.0, 200.0), track_id=1)

        # Set velocity manually
        state.x[2, 0] = 5.0  # vx = 5
        state.x[3, 0] = -3.0  # vy = -3

        # Predict next state
        predicted = kf.predict(state)

        # Check position updated based on velocity
        assert predicted.x[0, 0] == pytest.approx(105.0, rel=0.1)  # x + vx
        assert predicted.x[1, 0] == pytest.approx(197.0, rel=0.1)  # y + vy

    def test_update(self):
        """Test Kalman filter update (correction) step."""
        kf = AdvancedKalmanFilter(dt=1.0)
        state = kf.initialize((100.0, 200.0), track_id=1)

        # Predict
        predicted = kf.predict(state)

        # Update with measurement
        measurement = (105.0, 205.0)
        updated = kf.update(predicted, measurement)

        # Updated position should be between prediction and measurement
        assert updated.x[0, 0] >= 100.0
        assert updated.x[0, 0] <= 110.0
        assert updated.x[1, 0] >= 200.0
        assert updated.x[1, 0] <= 210.0

    def test_batch_predict(self):
        """Test batch prediction for multiple tracks."""
        kf = AdvancedKalmanFilter(dt=1.0)

        # Create multiple states
        states = [
            kf.initialize((100.0, 200.0), track_id=1),
            kf.initialize((150.0, 250.0), track_id=2),
            kf.initialize((200.0, 300.0), track_id=3),
        ]

        # Batch predict
        predicted_states = kf.predict_batch(states)

        assert len(predicted_states) == 3
        assert all(isinstance(s, KalmanState) for s in predicted_states)
        assert [s.track_id for s in predicted_states] == [1, 2, 3]

    def test_predict_n_steps(self):
        """Test multi-step prediction for occlusion handling."""
        kf = AdvancedKalmanFilter(dt=1.0)
        state = kf.initialize((100.0, 200.0), track_id=1)

        # Set velocity
        state.x[2, 0] = 5.0
        state.x[3, 0] = -3.0

        # Predict 10 steps ahead
        predictions = kf.predict_n_steps(state, n=10)

        assert len(predictions) == 10
        # First prediction should be roughly (105, 197)
        assert predictions[0][0] == pytest.approx(105.0, rel=0.2)
        # Last prediction should be further away
        assert predictions[9][0] > 120.0


class TestOpticalFlow:
    """Test suite for optical flow computation."""

    def test_cuda_optical_flow_init(self):
        """Test CUDA optical flow initialization."""
        flow = CUDAOpticalFlow(pyr_scale=0.5, levels=3, winsize=15)

        assert flow.pyr_scale == 0.5
        assert flow.levels == 3
        assert flow.winsize == 15
        assert flow.prev_frame is None

    def test_compute_flow_first_frame(self):
        """Test optical flow returns None on first frame."""
        flow = CUDAOpticalFlow()

        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        result = flow.compute_flow(frame)

        # First frame should return None
        assert result is None
        assert flow.prev_frame is not None

    def test_compute_flow_second_frame(self):
        """Test optical flow computation on second frame."""
        flow = CUDAOpticalFlow(use_cuda=False)  # Force CPU for testing

        # Create two frames with slight motion
        frame1 = np.zeros((480, 640), dtype=np.uint8)
        frame1[200:300, 200:300] = 255  # White square

        frame2 = np.zeros((480, 640), dtype=np.uint8)
        frame2[210:310, 210:310] = 255  # Moved square

        flow.compute_flow(frame1)
        flow_field = flow.compute_flow(frame2)

        assert flow_field is not None
        assert flow_field.shape == (480, 640, 2)  # H x W x 2

    def test_get_flow_at_points(self):
        """Test extracting flow at specific points."""
        flow_obj = CUDAOpticalFlow()

        # Create dummy flow field
        flow_field = np.random.randn(480, 640, 2).astype(np.float32)

        # Extract flow at points
        points = [(100, 100), (200, 200), (300, 300)]
        flow_vectors = flow_obj.get_flow_at_points(flow_field, points)

        assert len(flow_vectors) == 3
        assert all(isinstance(fv, FlowVector) for fv in flow_vectors)

    def test_get_average_flow_in_bbox(self):
        """Test average flow in bounding box."""
        flow_obj = CUDAOpticalFlow()

        # Create flow field with constant flow
        flow_field = np.ones((480, 640, 2), dtype=np.float32) * 5.0

        # Get average flow in bbox
        bbox = (100, 100, 200, 200)
        avg_flow = flow_obj.get_average_flow_in_bbox(flow_field, bbox)

        assert avg_flow is not None
        assert avg_flow[0] == pytest.approx(5.0, rel=0.1)
        assert avg_flow[1] == pytest.approx(5.0, rel=0.1)

    def test_compute_flow_magnitude(self):
        """Test flow magnitude computation."""
        flow_obj = CUDAOpticalFlow()

        # Create flow field
        flow_field = np.zeros((480, 640, 2), dtype=np.float32)
        flow_field[:, :, 0] = 3.0  # dx = 3
        flow_field[:, :, 1] = 4.0  # dy = 4

        magnitude = flow_obj.compute_flow_magnitude(flow_field)

        # Magnitude should be sqrt(3^2 + 4^2) = 5
        assert magnitude.shape == (480, 640)
        assert magnitude[0, 0] == pytest.approx(5.0, rel=0.01)


class TestImprovedTracker:
    """Test suite for Improved Tracker."""

    def test_initialization(self):
        """Test improved tracker initialization."""
        tracker = ImprovedTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            use_optical_flow=False,  # Disable for simple test
            use_kalman=True,
        )

        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert len(tracker.improved_tracks) == 0

    def test_create_detection(self):
        """Test detection creation."""
        det = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.85,
            class_id=0,
            class_name="test_object",
            frame_id=1,
            timestamp=0.033,
        )

        assert det.center() == (150, 150)
        assert det.width() == 100
        assert det.height() == 100
        assert det.area() == 10000

    def test_initialize_tracker(self):
        """Test tracker initialization with detections."""
        tracker = ImprovedTracker(use_optical_flow=False, use_kalman=True)

        detections = [
            Detection((100, 100, 200, 200), 0.9, 0, "obj1", 0, 0.0),
            Detection((300, 300, 400, 400), 0.85, 0, "obj2", 0, 0.0),
        ]

        tracker.initialize(detections)

        assert len(tracker.improved_tracks) == 2
        assert tracker.improved_tracks[0].status == "tentative"
        assert tracker.improved_tracks[1].status == "tentative"

    def test_update_tracker(self):
        """Test tracker update with new detections."""
        tracker = ImprovedTracker(
            use_optical_flow=False, use_kalman=True, min_hits=2  # Faster confirmation for test
        )

        # Initialize with first detection
        initial_det = [Detection((100, 100, 200, 200), 0.9, 0, "obj", 0, 0.0)]
        tracker.initialize(initial_det)

        # Update with nearby detection
        new_det = [Detection((105, 105, 205, 205), 0.88, 0, "obj", 1, 0.033)]
        tracks = tracker.update(new_det)

        assert len(tracks) > 0
        assert tracks[0].hits == 2

    def test_track_confirmation(self):
        """Test track confirmation after min_hits."""
        tracker = ImprovedTracker(use_optical_flow=False, use_kalman=True, min_hits=3)

        # Initialize
        det1 = [Detection((100, 100, 200, 200), 0.9, 0, "obj", 0, 0.0)]
        tracker.initialize(det1)

        # Update multiple times
        det2 = [Detection((105, 105, 205, 205), 0.88, 0, "obj", 1, 0.033)]
        tracker.update(det2)

        det3 = [Detection((110, 110, 210, 210), 0.87, 0, "obj", 2, 0.066)]
        tracks = tracker.update(det3)

        # Should be confirmed after 3 hits
        assert tracker.improved_tracks[0].status == "confirmed"
        assert tracker.improved_tracks[0].hits == 3

    def test_funscript_generation(self):
        """Test funscript data generation."""
        tracker = ImprovedTracker(use_optical_flow=False, use_kalman=False)

        # Create track with multiple positions
        track = ImprovedTrack(
            track_id=1,
            kalman_state=KalmanState(x=np.zeros((6, 1)), P=np.eye(6), track_id=1),
            status="confirmed",
        )

        # Add positions and detections
        for i in range(10):
            y = 500 + i * 10
            det = Detection((500, y, 600, y + 100), 0.9, 0, "obj", i, i * 0.033)
            track.detections.append(det)
            track.positions.append((550.0, float(y + 50)))
            track.confidences.append(0.9)

        tracker.improved_tracks.append(track)

        # Generate funscript
        funscript = tracker.get_funscript_data(
            track_id=1, axis="vertical", frame_height=1080, fps=30.0
        )

        assert funscript is not None
        assert len(funscript.actions) == 10
        assert funscript.metadata["tracker"] == "ImprovedTracker"
        assert funscript.metadata["track_id"] == 1

    def test_get_stats(self):
        """Test tracker statistics."""
        tracker = ImprovedTracker(use_optical_flow=True, use_kalman=True)

        stats = tracker.get_stats()

        assert "frame_count" in stats
        assert "active_tracks" in stats
        assert "confirmed_tracks" in stats
        assert "fps" in stats
        assert "kalman_enabled" in stats
        assert "optical_flow_enabled" in stats


# Benchmark tests
class TestBenchmarks:
    """Benchmark tests for performance validation."""

    def test_tracker_fps_benchmark(self):
        """Benchmark tracker FPS on synthetic data."""
        tracker = ImprovedTracker(
            use_optical_flow=False, use_kalman=True  # Disable for consistent timing
        )

        # Create synthetic detections
        detections = [
            Detection(
                (100 + i * 10, 100 + i * 5, 200 + i * 10, 200 + i * 5), 0.9, 0, f"obj{i}", 0, 0.0
            )
            for i in range(5)
        ]

        tracker.initialize(detections)

        import time

        start_time = time.time()
        num_frames = 100

        # Run tracking for N frames
        for frame_idx in range(num_frames):
            # Update detections slightly
            updated_dets = [
                Detection(
                    (
                        100 + i * 10 + frame_idx,
                        100 + i * 5 + frame_idx,
                        200 + i * 10 + frame_idx,
                        200 + i * 5 + frame_idx,
                    ),
                    0.9,
                    0,
                    f"obj{i}",
                    frame_idx,
                    frame_idx * 0.033,
                )
                for i in range(5)
            ]
            tracker.update(updated_dets)

        elapsed = time.time() - start_time
        fps = num_frames / elapsed

        print(f"\nTracker FPS: {fps:.2f}")
        # Should achieve >50 FPS on CPU without optical flow
        assert fps > 50.0, f"Tracker too slow: {fps:.2f} FPS"

    def test_kalman_prediction_speed(self):
        """Benchmark Kalman filter prediction speed."""
        kf = AdvancedKalmanFilter(dt=1.0, use_gpu=False)  # CPU test

        # Create multiple states
        num_tracks = 100
        states = [
            kf.initialize((float(i * 10), float(i * 20)), track_id=i) for i in range(num_tracks)
        ]

        import time

        start_time = time.time()
        iterations = 1000

        for _ in range(iterations):
            kf.predict_batch(states)

        elapsed = time.time() - start_time
        predictions_per_sec = (num_tracks * iterations) / elapsed

        print(f"\nKalman predictions/sec: {predictions_per_sec:.0f}")
        # Should achieve >10k predictions/sec on CPU
        assert predictions_per_sec > 10000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
