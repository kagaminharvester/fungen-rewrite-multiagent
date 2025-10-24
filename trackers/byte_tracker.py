"""
ByteTrack: Simple and fast multi-object tracking algorithm.

ByteTrack is a fast baseline tracker that uses IoU matching and Kalman filtering
for motion prediction. Target: 120+ FPS, <50ms latency.

Reference: https://arxiv.org/abs/2110.06864
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from trackers.base_tracker import BaseTracker, Detection, FunscriptAction, FunscriptData, Track


@dataclass
class KalmanTrack:
    """Track with Kalman filter for motion prediction.

    State vector: [x, y, vx, vy]
    - x, y: Center position
    - vx, vy: Velocity
    """

    track_id: int
    kalman_filter: Optional[Any] = None  # cv2.KalmanFilter
    state: np.ndarray = field(default_factory=lambda: np.zeros((4, 1), dtype=np.float32))
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    status: str = "tentative"  # tentative, confirmed, lost
    detections: List[Detection] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    class_id: int = -1
    class_name: str = ""

    def to_track(self) -> Track:
        """Convert to base Track object.

        Returns:
            Track object with full history
        """
        positions = [(int(d.center()[0]), int(d.center()[1])) for d in self.detections]
        velocities = []

        # Calculate velocities from state if Kalman filter is available
        if len(positions) > 1:
            for i in range(1, len(positions)):
                vx = positions[i][0] - positions[i - 1][0]
                vy = positions[i][1] - positions[i - 1][1]
                velocities.append((float(vx), float(vy)))
        else:
            velocities = [(0.0, 0.0)]

        track = Track(
            track_id=self.track_id,
            detections=self.detections.copy(),
            positions=positions,
            velocities=velocities,
            confidences=self.confidences.copy(),
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            state=self.status,
            class_id=self.class_id,
            class_name=self.class_name,
        )

        return track


class ByteTracker(BaseTracker):
    """ByteTrack implementation with Kalman filtering.

    ByteTrack uses a two-stage matching process:
    1. Match high-confidence detections to tracks
    2. Match low-confidence detections to remaining tracks

    This allows recovery from temporary occlusions while maintaining speed.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1,
        use_kalman: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize ByteTracker.

        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum consecutive detections to confirm track
            iou_threshold: IoU threshold for matching
            high_threshold: Confidence threshold for high-confidence detections
            low_threshold: Confidence threshold for low-confidence detections
            use_kalman: Whether to use Kalman filter for prediction
            **kwargs: Additional parameters
        """
        super().__init__(max_age, min_hits, iou_threshold, **kwargs)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.use_kalman = use_kalman and CV2_AVAILABLE
        self.kalman_tracks: List[KalmanTrack] = []
        self.frame_times: List[float] = []  # For FPS calculation

    def _create_kalman_filter(self) -> Optional[Any]:
        """Create a Kalman filter for 2D tracking.

        State: [x, y, vx, vy]
        Measurement: [x, y]

        Returns:
            cv2.KalmanFilter or None if cv2 not available
        """
        if not self.use_kalman:
            return None

        # 4 state variables (x, y, vx, vy), 2 measurements (x, y)
        kf = cv2.KalmanFilter(4, 2)

        # State transition matrix (A)
        # x' = x + vx
        # y' = y + vy
        # vx' = vx
        # vy' = vy
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )

        # Measurement matrix (H)
        # We only measure x and y
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

        # Process noise covariance (Q)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Measurement noise covariance (R)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

        # Error covariance (P)
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        return kf

    def initialize(self, detections: List[Detection]) -> None:
        """Initialize tracker with first frame detections.

        Args:
            detections: List of detections from first frame
        """
        self.kalman_tracks = []
        self.frame_count = 0

        # Create initial tracks from high-confidence detections
        for det in detections:
            if det.confidence >= self.high_threshold:
                self._create_new_track(det)

    def _create_new_track(self, detection: Detection) -> KalmanTrack:
        """Create a new track from detection.

        Args:
            detection: Detection to create track from

        Returns:
            New KalmanTrack object
        """
        track = KalmanTrack(
            track_id=self.next_track_id,
            age=1,
            hits=1,
            time_since_update=0,
            status="tentative",
            class_id=detection.class_id,
            class_name=detection.class_name,
        )

        # Initialize Kalman filter
        if self.use_kalman:
            kf = self._create_kalman_filter()
            if kf is not None:
                cx, cy = detection.center()
                kf.statePost = np.array([cx, cy, 0, 0], dtype=np.float32).reshape(4, 1)
                track.kalman_filter = kf
                track.state = kf.statePost.copy()

        track.detections.append(detection)
        track.confidences.append(detection.confidence)

        self.kalman_tracks.append(track)
        self.next_track_id += 1

        return track

    def _predict_tracks(self) -> None:
        """Predict next state for all tracks using Kalman filter."""
        for track in self.kalman_tracks:
            if self.use_kalman and track.kalman_filter is not None:
                prediction = track.kalman_filter.predict()
                track.state = prediction.copy()

    def _calculate_iou_matrix(
        self, tracks: List[KalmanTrack], detections: List[Detection]
    ) -> np.ndarray:
        """Calculate IoU matrix between tracks and detections.

        Args:
            tracks: List of tracks
            detections: List of detections

        Returns:
            IoU matrix (N_tracks x N_detections)
        """
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for t_idx, track in enumerate(tracks):
            # Get predicted bbox from Kalman state or last detection
            if track.detections:
                last_det = track.detections[-1]
                track_bbox = last_det.bbox
            else:
                continue

            for d_idx, detection in enumerate(detections):
                iou = self.calculate_iou(track_bbox, detection.bbox)
                iou_matrix[t_idx, d_idx] = iou

        return iou_matrix

    def _match_tracks_detections(
        self, tracks: List[KalmanTrack], detections: List[Detection], iou_threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match tracks to detections using Hungarian algorithm.

        Args:
            tracks: List of tracks
            detections: List of detections
            iou_threshold: IoU threshold for matching

        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
            - matches: List of (track_idx, detection_idx) pairs
            - unmatched_tracks: List of track indices
            - unmatched_detections: List of detection indices
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        iou_matrix = self._calculate_iou_matrix(tracks, detections)

        # Simple greedy matching (fast alternative to Hungarian)
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))

        # Sort by IoU score (highest first)
        candidates = []
        for t_idx in range(len(tracks)):
            for d_idx in range(len(detections)):
                if iou_matrix[t_idx, d_idx] >= iou_threshold:
                    candidates.append((iou_matrix[t_idx, d_idx], t_idx, d_idx))

        candidates.sort(reverse=True)

        matched_tracks = set()
        matched_detections = set()

        for iou_score, t_idx, d_idx in candidates:
            if t_idx not in matched_tracks and d_idx not in matched_detections:
                matches.append((t_idx, d_idx))
                matched_tracks.add(t_idx)
                matched_detections.add(d_idx)

        unmatched_tracks = [t for t in unmatched_tracks if t not in matched_tracks]
        unmatched_detections = [d for d in unmatched_detections if d not in matched_detections]

        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with new frame detections.

        This implements the two-stage ByteTrack matching:
        1. Match high-confidence detections to tracks
        2. Match low-confidence detections to remaining tracks

        Args:
            detections: List of detections from current frame

        Returns:
            List of updated tracks
        """
        start_time = time.time()

        self.frame_count += 1

        # Predict next state for all tracks
        self._predict_tracks()

        # Separate high and low confidence detections
        high_detections = [d for d in detections if d.confidence >= self.high_threshold]
        low_detections = [
            d for d in detections if self.low_threshold <= d.confidence < self.high_threshold
        ]

        # First stage: Match high-confidence detections to all active tracks (confirmed + tentative)
        all_active_tracks = [
            t for t in self.kalman_tracks if t.status in ("confirmed", "tentative")
        ]
        matches_high, unmatched_tracks_high, unmatched_detections_high = (
            self._match_tracks_detections(all_active_tracks, high_detections, self.iou_threshold)
        )

        # Update matched tracks
        for track_idx, det_idx in matches_high:
            track = all_active_tracks[track_idx]
            detection = high_detections[det_idx]
            self._update_track(track, detection)

        # Second stage: Match remaining tracks with low-confidence detections
        remaining_tracks = [all_active_tracks[i] for i in unmatched_tracks_high]

        # Match with low-confidence detections
        matches_low, unmatched_tracks_low, unmatched_detections_low = self._match_tracks_detections(
            remaining_tracks, low_detections, self.iou_threshold
        )

        # Update matched tracks with low-confidence detections
        for track_idx, det_idx in matches_low:
            track = remaining_tracks[track_idx]
            detection = low_detections[det_idx]
            self._update_track(track, detection)

        # Create new tracks from unmatched high-confidence detections
        for det_idx in unmatched_detections_high:
            detection = high_detections[det_idx]
            self._create_new_track(detection)

        # Mark unmatched tracks as missed
        all_matched_track_ids = set()
        for track_idx, _ in matches_high:
            all_matched_track_ids.add(all_active_tracks[track_idx].track_id)
        for track_idx, _ in matches_low:
            all_matched_track_ids.add(remaining_tracks[track_idx].track_id)

        for track in self.kalman_tracks:
            if track.track_id not in all_matched_track_ids:
                track.time_since_update += 1
                track.age += 1

                # Mark as lost if too old
                if track.time_since_update > self.max_age:
                    track.status = "lost"

        # Remove lost tracks
        self.kalman_tracks = [t for t in self.kalman_tracks if t.status != "lost"]

        # Convert to Track objects
        tracks = [kt.to_track() for kt in self.kalman_tracks]

        # Update frame timing for FPS calculation
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)

        return tracks

    def _update_track(self, track: KalmanTrack, detection: Detection) -> None:
        """Update track with new detection.

        Args:
            track: Track to update
            detection: New detection
        """
        # Update Kalman filter
        if self.use_kalman and track.kalman_filter is not None:
            cx, cy = detection.center()
            measurement = np.array([cx, cy], dtype=np.float32).reshape(2, 1)
            track.kalman_filter.correct(measurement)
            track.state = track.kalman_filter.statePost.copy()

        # Update track state
        track.detections.append(detection)
        track.confidences.append(detection.confidence)
        track.hits += 1
        track.time_since_update = 0
        track.age += 1

        # Confirm track if enough hits
        if track.status == "tentative" and track.hits >= self.min_hits:
            track.status = "confirmed"

    def get_funscript_data(
        self,
        track_id: Optional[int] = None,
        axis: str = "vertical",
        frame_height: int = 1080,
        fps: float = 30.0,
        smooth: bool = True,
        smooth_window: int = 5,
    ) -> FunscriptData:
        """Convert tracking data to funscript format.

        Args:
            track_id: Specific track to convert (None for primary track)
            axis: Axis to track ("vertical" or "horizontal")
            frame_height: Video frame height for normalization
            fps: Video frame rate for timestamp calculation
            smooth: Whether to apply smoothing to positions
            smooth_window: Window size for smoothing (odd number)

        Returns:
            FunscriptData object ready for serialization
        """
        # Find target track
        target_track = None
        if track_id is not None:
            target_track = next((kt for kt in self.kalman_tracks if kt.track_id == track_id), None)
        else:
            # Get primary track (longest confirmed track)
            confirmed = [kt for kt in self.kalman_tracks if kt.status == "confirmed"]
            if confirmed:
                target_track = max(confirmed, key=lambda t: t.hits)

        if target_track is None or not target_track.detections:
            return FunscriptData(actions=[])

        # Extract positions from detections
        positions = []
        timestamps = []

        for det in target_track.detections:
            cx, cy = det.center()

            if axis == "vertical":
                pos = cy
            elif axis == "horizontal":
                pos = cx
            else:
                pos = cy

            # Normalize to 0-100 range
            normalized_pos = self.normalize_position(pos, frame_height, invert=False)

            # Calculate timestamp in milliseconds
            timestamp = int((det.frame_id / fps) * 1000)

            positions.append(normalized_pos)
            timestamps.append(timestamp)

        # Apply smoothing if requested
        if smooth and len(positions) > smooth_window:
            positions = self._smooth_positions(positions, smooth_window)

        # Create funscript actions
        actions = [FunscriptAction(at=ts, pos=pos) for ts, pos in zip(timestamps, positions)]

        # Add metadata
        metadata = {
            "tracker": "ByteTrack",
            "track_id": target_track.track_id,
            "axis": axis,
            "frame_height": frame_height,
            "fps": fps,
            "total_frames": len(actions),
            "average_confidence": float(np.mean(target_track.confidences)),
            "kalman_enabled": self.use_kalman,
        }

        return FunscriptData(
            version="1.0", inverted=False, range=90, actions=actions, metadata=metadata
        )

    def _smooth_positions(self, positions: List[int], window: int) -> List[int]:
        """Apply moving average smoothing to positions.

        Args:
            positions: List of position values
            window: Window size (odd number recommended)

        Returns:
            Smoothed positions
        """
        if len(positions) < window:
            return positions

        smoothed = []
        half_window = window // 2

        for i in range(len(positions)):
            start = max(0, i - half_window)
            end = min(len(positions), i + half_window + 1)
            window_vals = positions[start:end]
            smoothed_val = int(np.mean(window_vals))
            smoothed.append(smoothed_val)

        return smoothed

    def get_fps(self) -> float:
        """Calculate current tracking FPS.

        Returns:
            Average FPS over last 30 frames
        """
        if not self.frame_times:
            return 0.0

        avg_time = np.mean(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics.

        Returns:
            Dictionary with tracking statistics
        """
        return {
            "frame_count": self.frame_count,
            "active_tracks": len(self.kalman_tracks),
            "confirmed_tracks": len([t for t in self.kalman_tracks if t.status == "confirmed"]),
            "tentative_tracks": len([t for t in self.kalman_tracks if t.status == "tentative"]),
            "fps": self.get_fps(),
            "kalman_enabled": self.use_kalman,
            "high_threshold": self.high_threshold,
            "low_threshold": self.low_threshold,
            "iou_threshold": self.iou_threshold,
        }
