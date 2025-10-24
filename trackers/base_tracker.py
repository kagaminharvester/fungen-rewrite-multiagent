"""
Abstract base class for all tracking algorithms.

This module defines the interface that all tracking implementations must follow,
ensuring consistency across different tracking algorithms (ByteTrack, BoT-SORT, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    """Single object detection from YOLO model.

    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2) in pixels
        confidence: Detection confidence score (0.0-1.0)
        class_id: Integer class identifier
        class_name: Human-readable class name (e.g., "penis", "hand", "mouth")
        frame_id: Frame number where detection occurred
        timestamp: Video timestamp in milliseconds
    """

    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    frame_id: int = 0
    timestamp: float = 0.0

    def center(self) -> Tuple[int, int]:
        """Calculate center point of bounding box.

        Returns:
            (x, y) center coordinates
        """
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def area(self) -> int:
        """Calculate bounding box area.

        Returns:
            Area in square pixels
        """
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def width(self) -> int:
        """Calculate bounding box width."""
        return self.bbox[2] - self.bbox[0]

    def height(self) -> int:
        """Calculate bounding box height."""
        return self.bbox[3] - self.bbox[1]


@dataclass
class Track:
    """Tracked object across multiple frames.

    Attributes:
        track_id: Unique identifier for this track
        detections: History of detections for this track
        positions: Center positions (x, y) for each frame
        velocities: Velocity vectors (vx, vy) from motion prediction
        confidences: Detection confidence scores over time
        age: Number of frames this track has existed
        hits: Number of consecutive frames with successful detection
        time_since_update: Frames since last successful detection update
        state: Tracking state (e.g., "tentative", "confirmed", "lost")
        class_id: Object class ID
        class_name: Object class name
    """

    track_id: int
    detections: List[Detection] = field(default_factory=list)
    positions: List[Tuple[int, int]] = field(default_factory=list)
    velocities: List[Tuple[float, float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: str = "tentative"
    class_id: int = -1
    class_name: str = ""

    def update(self, detection: Detection) -> None:
        """Update track with new detection.

        Args:
            detection: New detection to add to track
        """
        self.detections.append(detection)
        self.positions.append(detection.center())
        self.confidences.append(detection.confidence)
        self.hits += 1
        self.time_since_update = 0
        self.age += 1

        if self.state == "tentative" and self.hits >= 3:
            self.state = "confirmed"

        # Update class information
        if not self.class_name:
            self.class_id = detection.class_id
            self.class_name = detection.class_name

    def mark_missed(self) -> None:
        """Mark track as missed (no detection in current frame)."""
        self.time_since_update += 1
        self.age += 1

        if self.time_since_update > 30:
            self.state = "lost"

    def get_last_position(self) -> Optional[Tuple[int, int]]:
        """Get the most recent position.

        Returns:
            Last (x, y) position or None if no positions
        """
        return self.positions[-1] if self.positions else None

    def get_last_detection(self) -> Optional[Detection]:
        """Get the most recent detection.

        Returns:
            Last detection or None if no detections
        """
        return self.detections[-1] if self.detections else None

    def get_average_confidence(self) -> float:
        """Calculate average confidence over track lifetime.

        Returns:
            Average confidence score
        """
        return np.mean(self.confidences) if self.confidences else 0.0

    def predict_position(self) -> Tuple[int, int]:
        """Predict next position using velocity.

        Returns:
            Predicted (x, y) position
        """
        if not self.positions or not self.velocities:
            return self.get_last_position() or (0, 0)

        last_pos = self.positions[-1]
        last_vel = self.velocities[-1] if self.velocities else (0, 0)

        pred_x = int(last_pos[0] + last_vel[0])
        pred_y = int(last_pos[1] + last_vel[1])

        return (pred_x, pred_y)


@dataclass
class FunscriptAction:
    """Single funscript action point.

    Attributes:
        at: Timestamp in milliseconds
        pos: Position value (0-100)
    """

    at: int
    pos: int


@dataclass
class FunscriptData:
    """Complete funscript data structure.

    Attributes:
        version: Funscript format version
        inverted: Whether positions are inverted
        range: Position range (default 90)
        actions: List of action points
        metadata: Additional metadata
    """

    version: str = "1.0"
    inverted: bool = False
    range: int = 90
    actions: List[FunscriptAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            "version": self.version,
            "inverted": self.inverted,
            "range": self.range,
            "actions": [{"at": a.at, "pos": a.pos} for a in self.actions],
            "metadata": self.metadata,
        }


class BaseTracker(ABC):
    """Abstract base class for all tracking algorithms.

    This class defines the interface that all tracker implementations must follow.
    Subclasses must implement initialize(), update(), and get_funscript_data().
    """

    def __init__(
        self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3, **kwargs: Any
    ) -> None:
        """Initialize base tracker.

        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum consecutive detections to confirm track
            iou_threshold: Intersection over Union threshold for matching
            **kwargs: Additional tracker-specific parameters
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.frame_count = 0

    @abstractmethod
    def initialize(self, detections: List[Detection]) -> None:
        """Initialize tracker with first frame detections.

        Args:
            detections: List of detections from first frame
        """
        pass

    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with new frame detections.

        Args:
            detections: List of detections from current frame

        Returns:
            List of updated tracks
        """
        pass

    @abstractmethod
    def get_funscript_data(
        self, track_id: Optional[int] = None, axis: str = "vertical"
    ) -> FunscriptData:
        """Convert tracking data to funscript format.

        Args:
            track_id: Specific track to convert (None for primary track)
            axis: Axis to track ("vertical" or "horizontal")

        Returns:
            FunscriptData object ready for serialization
        """
        pass

    def get_active_tracks(self) -> List[Track]:
        """Get currently active (confirmed and not lost) tracks.

        Returns:
            List of active tracks
        """
        return [
            track
            for track in self.tracks
            if track.state in ("confirmed", "tentative") and track.time_since_update < self.max_age
        ]

    def get_primary_track(self) -> Optional[Track]:
        """Get the primary track (longest, highest confidence).

        Returns:
            Primary track or None if no tracks exist
        """
        active_tracks = self.get_active_tracks()
        if not active_tracks:
            return None

        # Sort by track length (number of hits), then by average confidence
        return max(active_tracks, key=lambda t: (t.hits, t.get_average_confidence()))

    def calculate_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union between two bounding boxes.

        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)

        Returns:
            IoU score (0.0-1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def normalize_position(self, position: int, frame_height: int, invert: bool = False) -> int:
        """Normalize position to funscript range (0-100).

        Args:
            position: Y-coordinate position in pixels
            frame_height: Video frame height in pixels
            invert: Whether to invert the position

        Returns:
            Normalized position (0-100)
        """
        # Convert pixel position to 0-100 range
        normalized = int((position / frame_height) * 100)
        normalized = max(0, min(100, normalized))

        if invert:
            normalized = 100 - normalized

        return normalized

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0
