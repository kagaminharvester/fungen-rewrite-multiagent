"""
Example usage of ByteTracker for object tracking.

This script demonstrates various tracking scenarios:
1. Simple single-object tracking
2. Multi-object tracking
3. Handling occlusions
4. Funscript generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math

from trackers import ByteTracker, Detection


def example_1_simple_tracking():
    """Example 1: Simple single-object tracking."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Single-Object Tracking")
    print("=" * 60)

    # Create tracker
    tracker = ByteTracker(use_kalman=True)

    # Simulate object moving horizontally
    print("\nTracking object moving right...")
    detections_sequence = []
    for frame_id in range(100):
        det = Detection(
            bbox=(100 + frame_id * 2, 100, 200 + frame_id * 2, 200),
            confidence=0.9,
            class_id=0,
            class_name="object",
            frame_id=frame_id,
            timestamp=frame_id * 0.033,  # 30 FPS
        )
        detections_sequence.append([det])

    # Initialize with first frame
    tracker.initialize(detections_sequence[0])

    # Process remaining frames
    for dets in detections_sequence[1:]:
        tracks = tracker.update(dets)

    print(f"✓ Tracking completed")
    print(f"  FPS: {tracker.get_fps():.2f}")
    print(f"  Active tracks: {len(tracker.get_active_tracks())}")

    # Get stats
    stats = tracker.get_stats()
    print(f"  Confirmed tracks: {stats['confirmed_tracks']}")
    print(f"  Frame count: {stats['frame_count']}")


def example_2_multi_object():
    """Example 2: Multi-object tracking."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Object Tracking")
    print("=" * 60)

    tracker = ByteTracker(use_kalman=True)

    # Create 3 objects with different motion patterns
    def create_detections(frame_id):
        return [
            # Object 1: moving right
            Detection(
                bbox=(100 + frame_id * 3, 100, 200 + frame_id * 3, 200),
                confidence=0.9,
                class_id=0,
                class_name="object1",
                frame_id=frame_id,
                timestamp=frame_id * 0.033,
            ),
            # Object 2: moving down
            Detection(
                bbox=(300, 100 + frame_id * 2, 400, 200 + frame_id * 2),
                confidence=0.85,
                class_id=0,
                class_name="object2",
                frame_id=frame_id,
                timestamp=frame_id * 0.033,
            ),
            # Object 3: stationary
            Detection(
                bbox=(500, 300, 600, 400),
                confidence=0.95,
                class_id=0,
                class_name="object3",
                frame_id=frame_id,
                timestamp=frame_id * 0.033,
            ),
        ]

    print("\nTracking 3 objects with different motion patterns...")

    # Initialize
    tracker.initialize(create_detections(0))

    # Process frames
    for frame_id in range(1, 50):
        tracks = tracker.update(create_detections(frame_id))

    print(f"✓ Multi-object tracking completed")
    print(f"  Active tracks: {len(tracker.get_active_tracks())}")

    for track in tracker.get_active_tracks():
        print(
            f"  Track {track.track_id}: "
            f"{track.hits} hits, "
            f"confidence: {track.get_average_confidence():.2f}, "
            f"positions: {len(track.positions)}"
        )


def example_3_occlusion():
    """Example 3: Handling occlusions."""
    print("\n" + "=" * 60)
    print("Example 3: Handling Occlusions")
    print("=" * 60)

    tracker = ByteTracker(
        max_age=30,  # Keep track for 30 frames without detection
        use_kalman=True,  # Use Kalman for prediction during occlusion
    )

    print("\nSimulating object occlusion (disappears for 10 frames)...")

    # Simulate object that disappears for 10 frames (frames 20-29)
    detections_sequence = []
    for frame_id in range(50):
        if 20 <= frame_id < 30:
            # Object occluded (no detection)
            detections_sequence.append([])
        else:
            det = Detection(
                bbox=(100 + frame_id * 2, 100, 200 + frame_id * 2, 200),
                confidence=0.9,
                class_id=0,
                class_name="object",
                frame_id=frame_id,
                timestamp=frame_id * 0.033,
            )
            detections_sequence.append([det])

    # Track
    tracker.initialize(detections_sequence[0])

    for frame_idx, dets in enumerate(detections_sequence[1:], start=1):
        tracks = tracker.update(dets)

        if frame_idx == 19:
            print(f"  Before occlusion - Track maintained: " f"{len(tracks) > 0}")
        elif frame_idx == 25:
            print(f"  During occlusion (no detection) - Track maintained: " f"{len(tracks) > 0}")
        elif frame_idx == 30:
            print(f"  After occlusion - Track maintained: " f"{len(tracks) > 0}")

    active_tracks = tracker.get_active_tracks()
    if active_tracks:
        track = active_tracks[0]
        print(f"\n✓ Track successfully maintained through occlusion")
        print(f"  Track ID: {track.track_id}")
        print(f"  Total hits: {track.hits}")
        print(f"  Track age: {track.age}")
    else:
        print("✗ Track lost during occlusion")


def example_4_funscript():
    """Example 4: Funscript generation."""
    print("\n" + "=" * 60)
    print("Example 4: Funscript Generation")
    print("=" * 60)

    tracker = ByteTracker(use_kalman=True)

    print("\nTracking object with sinusoidal vertical motion...")

    # Track object moving vertically (sinusoidal motion)
    detections_sequence = []
    for frame_id in range(60):
        # Sinusoidal motion (simulating up/down movement)
        y_pos = 500 + int(200 * math.sin(frame_id * 0.15))

        det = Detection(
            bbox=(400, y_pos, 500, y_pos + 100),
            confidence=0.9,
            class_id=0,
            class_name="target",
            frame_id=frame_id,
            timestamp=frame_id * 0.033,
        )
        detections_sequence.append([det])

    # Track
    tracker.initialize(detections_sequence[0])
    for dets in detections_sequence[1:]:
        tracker.update(dets)

    # Generate funscript
    print("\nGenerating funscript data...")
    funscript = tracker.get_funscript_data(
        frame_height=1080, fps=30.0, smooth=True, smooth_window=5
    )

    # Display funscript info
    print(f"✓ Funscript generated")
    print(f"  Actions: {len(funscript.actions)}")
    print(f"  Duration: {funscript.actions[-1].at}ms")
    print(f"  Tracker: {funscript.metadata.get('tracker')}")
    print(f"  Track ID: {funscript.metadata.get('track_id')}")

    # Show first few actions
    print("\n  First 5 actions:")
    for action in funscript.actions[:5]:
        print(f"    {action.at}ms -> position {action.pos}")

    # Save funscript (optional)
    output_path = Path(__file__).parent / "example_output.funscript"
    with open(output_path, "w") as f:
        json.dump(funscript.to_dict(), f, indent=2)

    print(f"\n  Saved to: {output_path}")


def example_5_varying_confidence():
    """Example 5: Two-stage matching with varying confidence."""
    print("\n" + "=" * 60)
    print("Example 5: Two-Stage Matching")
    print("=" * 60)

    tracker = ByteTracker(high_threshold=0.6, low_threshold=0.1, use_kalman=True)

    print("\nTracking object with varying detection confidence...")

    # Simulate object with varying confidence (high -> low -> high)
    detections_sequence = []
    for frame_id in range(50):
        # Confidence varies over time
        if 15 <= frame_id < 25:
            confidence = 0.3  # Low confidence period
        else:
            confidence = 0.9  # High confidence

        det = Detection(
            bbox=(100 + frame_id * 2, 100, 200 + frame_id * 2, 200),
            confidence=confidence,
            class_id=0,
            class_name="object",
            frame_id=frame_id,
            timestamp=frame_id * 0.033,
        )
        detections_sequence.append([det])

    # Track
    tracker.initialize(detections_sequence[0])

    track_maintained = True
    for frame_idx, dets in enumerate(detections_sequence[1:], start=1):
        tracks = tracker.update(dets)

        if frame_idx == 14:
            print(f"  Frame {frame_idx} (high confidence): " f"Tracks = {len(tracks)}")
        elif frame_idx == 20:
            print(f"  Frame {frame_idx} (low confidence): " f"Tracks = {len(tracks)}")
        elif frame_idx == 30:
            print(f"  Frame {frame_idx} (high confidence again): " f"Tracks = {len(tracks)}")

        if len(tracks) == 0:
            track_maintained = False

    if track_maintained:
        print(f"\n✓ Track maintained through low-confidence period")
        print(f"  Two-stage matching successfully handled confidence variation")
    else:
        print("\n✗ Track lost during low-confidence period")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ByteTracker Examples")
    print("=" * 60)

    examples = [
        example_1_simple_tracking,
        example_2_multi_object,
        example_3_occlusion,
        example_4_funscript,
        example_5_varying_confidence,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
