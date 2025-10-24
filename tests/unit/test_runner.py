"""
Simple test runner for ByteTracker tests (when pytest is not available).

This script runs basic functionality tests to verify the tracker implementation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time

from trackers.base_tracker import Detection, Track
from trackers.byte_tracker import ByteTracker


def test_detection_creation():
    """Test Detection creation and methods."""
    print("Testing Detection creation...")
    det = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.9,
        class_id=0,
        class_name="test_object",
        frame_id=0,
        timestamp=0.0,
    )

    assert det.center() == (150, 150), "Center calculation failed"
    assert det.area() == 10000, "Area calculation failed"
    assert det.width() == 100, "Width calculation failed"
    assert det.height() == 100, "Height calculation failed"
    print("  ✓ Detection tests passed")


def test_track_update():
    """Test Track update functionality."""
    print("Testing Track update...")
    track = Track(track_id=1)

    det = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.9,
        class_id=0,
        class_name="test",
        frame_id=0,
        timestamp=0.0,
    )

    track.update(det)

    assert track.hits == 1, "Track hits not updated"
    assert track.age == 1, "Track age not updated"
    assert len(track.detections) == 1, "Detection not added"
    print("  ✓ Track update tests passed")


def test_byte_tracker_init():
    """Test ByteTracker initialization."""
    print("Testing ByteTracker initialization...")

    tracker = ByteTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        high_threshold=0.6,
        low_threshold=0.1,
        use_kalman=True,
    )

    detections = [
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

    tracker.initialize(detections)

    assert len(tracker.kalman_tracks) == 2, "Tracks not created"
    assert tracker.next_track_id == 3, "Track ID counter not updated"
    print("  ✓ ByteTracker initialization tests passed")


def test_byte_tracker_update():
    """Test ByteTracker update with matching detection."""
    print("Testing ByteTracker update...")

    tracker = ByteTracker(use_kalman=True)

    det1 = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.9,
        class_id=0,
        class_name="object",
        frame_id=0,
        timestamp=0.0,
    )
    tracker.initialize([det1])

    det2 = Detection(
        bbox=(105, 105, 205, 205),
        confidence=0.85,
        class_id=0,
        class_name="object",
        frame_id=1,
        timestamp=0.033,
    )
    tracks = tracker.update([det2])

    assert len(tracks) == 1, "Track not maintained"
    assert tracks[0].hits == 2, "Track hits not updated"
    print("  ✓ ByteTracker update tests passed")


def test_iou_calculation():
    """Test IoU calculation."""
    print("Testing IoU calculation...")

    tracker = ByteTracker()

    # Perfect overlap
    iou = tracker.calculate_iou((0, 0, 100, 100), (0, 0, 100, 100))
    assert abs(iou - 1.0) < 0.01, "Perfect overlap IoU failed"

    # No overlap
    iou = tracker.calculate_iou((0, 0, 100, 100), (200, 200, 300, 300))
    assert iou == 0.0, "No overlap IoU failed"

    # Partial overlap
    iou = tracker.calculate_iou((0, 0, 100, 100), (50, 50, 150, 150))
    assert 0.1 < iou < 0.2, "Partial overlap IoU failed"

    print("  ✓ IoU calculation tests passed")


def test_two_stage_matching():
    """Test two-stage matching with high and low confidence."""
    print("Testing two-stage matching...")

    tracker = ByteTracker(high_threshold=0.6, low_threshold=0.1, use_kalman=True)

    # Initialize with high-confidence
    det1 = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.9,
        class_id=0,
        class_name="object",
        frame_id=0,
        timestamp=0.0,
    )
    tracker.initialize([det1])

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
        tracker.update([det])

    # Update with low-confidence detection
    det_low = Detection(
        bbox=(120, 100, 220, 200),
        confidence=0.4,
        class_id=0,
        class_name="object",
        frame_id=4,
        timestamp=0.133,
    )
    tracks = tracker.update([det_low])

    assert len(tracks) == 1, "Track not maintained with low confidence"
    assert tracks[0].hits == 5, "Low confidence detection not matched"
    print("  ✓ Two-stage matching tests passed")


def test_funscript_generation():
    """Test funscript data generation."""
    print("Testing funscript generation...")

    tracker = ByteTracker(use_kalman=True)

    # Create track with multiple detections
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

    tracker.initialize([detections[0]])
    for det in detections[1:]:
        tracker.update([det])

    # Generate funscript
    funscript = tracker.get_funscript_data(frame_height=1080, fps=30.0, smooth=False)

    assert len(funscript.actions) > 0, "No funscript actions generated"
    assert funscript.version == "1.0", "Incorrect version"
    assert "tracker" in funscript.metadata, "Missing metadata"
    print("  ✓ Funscript generation tests passed")


def test_performance():
    """Test tracker performance (target: 120+ FPS)."""
    print("Testing performance...")

    tracker = ByteTracker(use_kalman=True)

    det = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.9,
        class_id=0,
        class_name="object",
        frame_id=0,
        timestamp=0.0,
    )
    tracker.initialize([det])

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
        tracker.update([det])

    elapsed = time.time() - start_time
    fps = num_iterations / elapsed

    print(f"  ByteTracker FPS: {fps:.2f}")

    if fps > 120:
        print("  ✓ Performance target achieved (120+ FPS)")
    elif fps > 50:
        print(f"  ⚠ Performance acceptable but below target: {fps:.2f} FPS (target: 120+)")
    else:
        print(f"  ✗ Performance below acceptable: {fps:.2f} FPS")

    return fps


def test_multi_object_tracking():
    """Test tracking multiple objects simultaneously."""
    print("Testing multi-object tracking...")

    tracker = ByteTracker(use_kalman=True)

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

    tracker.initialize(initial_dets)

    # Update with 5 moving objects
    for frame_id in range(10):
        dets = []
        for i in range(5):
            det = Detection(
                bbox=(100 + i * 150 + frame_id * 5, 100, 200 + i * 150, 200),
                confidence=0.9,
                class_id=0,
                class_name=f"object{i}",
                frame_id=frame_id + 1,
                timestamp=float(frame_id + 1) * 0.033,
            )
            dets.append(det)

        tracks = tracker.update(dets)

    assert len(tracker.kalman_tracks) == 5, "Not all objects tracked"
    print("  ✓ Multi-object tracking tests passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ByteTracker Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_detection_creation,
        test_track_update,
        test_byte_tracker_init,
        test_byte_tracker_update,
        test_iou_calculation,
        test_two_stage_matching,
        test_funscript_generation,
        test_multi_object_tracking,
        test_performance,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
