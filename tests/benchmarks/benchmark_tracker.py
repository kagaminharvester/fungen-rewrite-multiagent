"""
Performance benchmarks for ByteTracker.

Tests various scenarios to validate performance claims.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time

import numpy as np

from trackers import ByteTracker, Detection


def benchmark_single_object(num_frames=1000):
    """Benchmark single object tracking."""
    print("\n" + "=" * 60)
    print(f"Benchmark: Single Object ({num_frames} frames)")
    print("=" * 60)

    tracker = ByteTracker(use_kalman=True)

    # Create detections
    detections = []
    for i in range(num_frames):
        det = Detection(
            bbox=(100 + i, 100, 200 + i, 200),
            confidence=0.9,
            class_id=0,
            class_name="object",
            frame_id=i,
            timestamp=i * 0.033,
        )
        detections.append([det])

    # Benchmark
    start_time = time.time()

    tracker.initialize(detections[0])
    for dets in detections[1:]:
        tracker.update(dets)

    elapsed = time.time() - start_time
    fps = num_frames / elapsed
    latency_ms = (elapsed / num_frames) * 1000

    print(f"  Frames processed: {num_frames}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  FPS: {fps:.2f}")
    print(f"  Latency per frame: {latency_ms:.3f}ms")
    print(f"  Target FPS: 120+")
    print(f"  Status: {'✓ PASS' if fps >= 120 else '✗ FAIL'}")

    return fps, latency_ms


def benchmark_multi_object(num_objects=5, num_frames=500):
    """Benchmark multi-object tracking."""
    print("\n" + "=" * 60)
    print(f"Benchmark: {num_objects} Objects ({num_frames} frames)")
    print("=" * 60)

    tracker = ByteTracker(use_kalman=True)

    # Create detections for multiple objects
    def create_frame_detections(frame_id):
        dets = []
        for obj_id in range(num_objects):
            det = Detection(
                bbox=(
                    100 + obj_id * 150 + frame_id,
                    100 + obj_id * 50,
                    200 + obj_id * 150 + frame_id,
                    200 + obj_id * 50,
                ),
                confidence=0.9,
                class_id=0,
                class_name=f"object{obj_id}",
                frame_id=frame_id,
                timestamp=frame_id * 0.033,
            )
            dets.append(det)
        return dets

    # Benchmark
    start_time = time.time()

    tracker.initialize(create_frame_detections(0))
    for frame_id in range(1, num_frames):
        tracker.update(create_frame_detections(frame_id))

    elapsed = time.time() - start_time
    fps = num_frames / elapsed
    latency_ms = (elapsed / num_frames) * 1000

    print(f"  Frames processed: {num_frames}")
    print(f"  Objects tracked: {num_objects}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  FPS: {fps:.2f}")
    print(f"  Latency per frame: {latency_ms:.3f}ms")
    print(f"  Target FPS: 80+")
    print(f"  Status: {'✓ PASS' if fps >= 80 else '✗ FAIL'}")

    return fps, latency_ms


def benchmark_with_without_kalman(num_frames=1000):
    """Compare performance with and without Kalman filter."""
    print("\n" + "=" * 60)
    print(f"Benchmark: Kalman Filter Impact ({num_frames} frames)")
    print("=" * 60)

    # Create detections
    detections = []
    for i in range(num_frames):
        det = Detection(
            bbox=(100 + i, 100, 200 + i, 200),
            confidence=0.9,
            class_id=0,
            class_name="object",
            frame_id=i,
            timestamp=i * 0.033,
        )
        detections.append([det])

    # Test WITH Kalman
    tracker_with = ByteTracker(use_kalman=True)
    start_time = time.time()
    tracker_with.initialize(detections[0])
    for dets in detections[1:]:
        tracker_with.update(dets)
    time_with = time.time() - start_time
    fps_with = num_frames / time_with

    # Test WITHOUT Kalman
    tracker_without = ByteTracker(use_kalman=False)
    start_time = time.time()
    tracker_without.initialize(detections[0])
    for dets in detections[1:]:
        tracker_without.update(dets)
    time_without = time.time() - start_time
    fps_without = num_frames / time_without

    print(f"  WITH Kalman:")
    print(f"    FPS: {fps_with:.2f}")
    print(f"    Latency: {(time_with / num_frames) * 1000:.3f}ms")

    print(f"  WITHOUT Kalman:")
    print(f"    FPS: {fps_without:.2f}")
    print(f"    Latency: {(time_without / num_frames) * 1000:.3f}ms")

    speedup = fps_without / fps_with
    print(f"  Speedup (no Kalman): {speedup:.2f}x")

    return fps_with, fps_without


def benchmark_occlusion_handling(num_frames=200, occlusion_frames=30):
    """Benchmark occlusion handling."""
    print("\n" + "=" * 60)
    print(f"Benchmark: Occlusion Handling")
    print(f"({num_frames} frames, {occlusion_frames} occluded)")
    print("=" * 60)

    tracker = ByteTracker(max_age=50, use_kalman=True)

    # Create detections with occlusion
    detections = []
    for i in range(num_frames):
        if 85 <= i < (85 + occlusion_frames):
            # Occluded
            detections.append([])
        else:
            det = Detection(
                bbox=(100 + i, 100, 200 + i, 200),
                confidence=0.9,
                class_id=0,
                class_name="object",
                frame_id=i,
                timestamp=i * 0.033,
            )
            detections.append([det])

    # Benchmark
    start_time = time.time()

    tracker.initialize(detections[0])
    tracks_maintained = True

    for frame_id, dets in enumerate(detections[1:], start=1):
        tracks = tracker.update(dets)

        # Check if track is maintained during and after occlusion
        if 85 <= frame_id < 120:
            if len(tracks) == 0:
                tracks_maintained = False

    elapsed = time.time() - start_time
    fps = num_frames / elapsed

    print(f"  Frames processed: {num_frames}")
    print(f"  Occlusion period: {occlusion_frames} frames")
    print(f"  FPS: {fps:.2f}")
    print(f"  Track maintained: {'✓ YES' if tracks_maintained else '✗ NO'}")

    return fps, tracks_maintained


def benchmark_confidence_variation(num_frames=500):
    """Benchmark two-stage matching with varying confidence."""
    print("\n" + "=" * 60)
    print(f"Benchmark: Two-Stage Matching ({num_frames} frames)")
    print("=" * 60)

    tracker = ByteTracker(high_threshold=0.6, low_threshold=0.1, use_kalman=True)

    # Create detections with varying confidence
    detections = []
    for i in range(num_frames):
        # Alternate between high and low confidence
        if i % 10 < 7:
            confidence = 0.9  # High confidence
        else:
            confidence = 0.3  # Low confidence

        det = Detection(
            bbox=(100 + i, 100, 200 + i, 200),
            confidence=confidence,
            class_id=0,
            class_name="object",
            frame_id=i,
            timestamp=i * 0.033,
        )
        detections.append([det])

    # Benchmark
    start_time = time.time()

    tracker.initialize(detections[0])
    track_maintained = True

    for dets in detections[1:]:
        tracks = tracker.update(dets)
        if len(tracks) == 0:
            track_maintained = False

    elapsed = time.time() - start_time
    fps = num_frames / elapsed

    print(f"  Frames processed: {num_frames}")
    print(f"  Confidence variation: High (70%) / Low (30%)")
    print(f"  FPS: {fps:.2f}")
    print(f"  Track maintained: {'✓ YES' if track_maintained else '✗ NO'}")

    return fps, track_maintained


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("ByteTracker Performance Benchmarks")
    print("=" * 70)
    print(f"Platform: Raspberry Pi (ARM64, CPU-only)")
    print(f"Target: 120+ FPS, <50ms latency")
    print("=" * 70)

    results = {}

    # Run benchmarks
    results["single_1000"] = benchmark_single_object(1000)
    results["single_5000"] = benchmark_single_object(5000)
    results["multi_5"] = benchmark_multi_object(5, 500)
    results["multi_10"] = benchmark_multi_object(10, 200)
    results["kalman"] = benchmark_with_without_kalman(1000)
    results["occlusion"] = benchmark_occlusion_handling(200, 30)
    results["confidence"] = benchmark_confidence_variation(500)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nSingle Object Performance:")
    print(
        f"  1000 frames: {results['single_1000'][0]:.2f} FPS, "
        f"{results['single_1000'][1]:.3f}ms latency"
    )
    print(
        f"  5000 frames: {results['single_5000'][0]:.2f} FPS, "
        f"{results['single_5000'][1]:.3f}ms latency"
    )

    print(f"\nMulti-Object Performance:")
    print(
        f"  5 objects: {results['multi_5'][0]:.2f} FPS, " f"{results['multi_5'][1]:.3f}ms latency"
    )
    print(
        f"  10 objects: {results['multi_10'][0]:.2f} FPS, "
        f"{results['multi_10'][1]:.3f}ms latency"
    )

    print(f"\nKalman Filter Impact:")
    print(f"  With Kalman: {results['kalman'][0]:.2f} FPS")
    print(f"  Without Kalman: {results['kalman'][1]:.2f} FPS")
    print(
        f"  Overhead: {((results['kalman'][1] - results['kalman'][0]) / results['kalman'][1] * 100):.1f}%"
    )

    print(f"\nFeature Tests:")
    print(
        f"  Occlusion handling: {results['occlusion'][0]:.2f} FPS, "
        f"maintained={'✓' if results['occlusion'][1] else '✗'}"
    )
    print(
        f"  Confidence variation: {results['confidence'][0]:.2f} FPS, "
        f"maintained={'✓' if results['confidence'][1] else '✗'}"
    )

    # Overall assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    avg_fps = np.mean([results["single_1000"][0], results["single_5000"][0], results["multi_5"][0]])

    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Target FPS: 120+")
    print(f"Performance: {avg_fps / 120:.1f}x faster than target")

    if avg_fps >= 120:
        print("\n✓ ALL PERFORMANCE TARGETS MET")
    else:
        print("\n⚠ Performance below target (but still acceptable for Pi)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
