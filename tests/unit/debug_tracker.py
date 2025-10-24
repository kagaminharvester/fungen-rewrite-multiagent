"""Debug script to understand tracker behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trackers.base_tracker import Detection
from trackers.byte_tracker import ByteTracker


def debug_simple_update():
    """Debug simple update case."""
    print("=" * 60)
    print("Debugging Simple Update")
    print("=" * 60)

    tracker = ByteTracker(use_kalman=True)

    det1 = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.9,
        class_id=0,
        class_name="object",
        frame_id=0,
        timestamp=0.0,
    )

    print(f"\n1. Initialize with detection: {det1.center()}, conf={det1.confidence}")
    tracker.initialize([det1])
    print(f"   Tracks created: {len(tracker.kalman_tracks)}")
    for track in tracker.kalman_tracks:
        print(f"   Track {track.track_id}: status={track.status}, hits={track.hits}")

    det2 = Detection(
        bbox=(105, 105, 205, 205),
        confidence=0.85,
        class_id=0,
        class_name="object",
        frame_id=1,
        timestamp=0.033,
    )

    print(f"\n2. Update with detection: {det2.center()}, conf={det2.confidence}")
    print(f"   IoU between detections: {tracker.calculate_iou(det1.bbox, det2.bbox):.3f}")

    tracks = tracker.update([det2])

    print(f"   Active tracks after update: {len(tracks)}")
    for track in tracks:
        print(
            f"   Track {track.track_id}: hits={track.hits}, time_since_update={track.time_since_update}"
        )
        print(f"   Positions: {track.positions}")

    print(f"\n   Kalman tracks: {len(tracker.kalman_tracks)}")
    for track in tracker.kalman_tracks:
        print(f"   Track {track.track_id}: status={track.status}, hits={track.hits}")


if __name__ == "__main__":
    debug_simple_update()
