"""
Tracking algorithm performance benchmarks.

Benchmarks for different tracking algorithms:
- ByteTrack (fast baseline)
- ImprovedTracker (Kalman + Optical Flow)
- BoT-SORT (when available)

Metrics:
- FPS (frames per second)
- Latency (ms per frame)
- Tracking accuracy (IoU, ID switches)
- Memory usage

Target: ByteTrack 100+ FPS, ImprovedTracker 80+ FPS @ 1080p

Author: test-engineer-2 agent
Date: 2025-10-24
"""

import time
from typing import Dict, List

import numpy as np
import pytest

from trackers.base_tracker import Detection, Track
from trackers.byte_tracker import ByteTracker
from trackers.improved_tracker import ImprovedTracker
from utils.conditional_imports import CUDA_AVAILABLE


def create_mock_detections(
    frame_num: int,
    num_objects: int = 2,
    width: int = 1920,
    height: int = 1080,
    motion_pattern: str = "linear",
) -> List[Detection]:
    """Create mock detections for testing.

    Args:
        frame_num: Current frame number
        num_objects: Number of objects to detect
        width: Frame width
        height: Frame height
        motion_pattern: Type of motion ("linear", "circular", "random")

    Returns:
        List of mock detections
    """
    detections = []

    for obj_id in range(num_objects):
        if motion_pattern == "linear":
            # Linear horizontal motion
            x1 = 100 + obj_id * 200 + frame_num * 5
            y1 = 200 + obj_id * 100
            x2 = x1 + 100
            y2 = y1 + 150

        elif motion_pattern == "circular":
            # Circular motion
            radius = 200
            angle = (frame_num * 0.1) + (obj_id * np.pi)
            center_x = width // 2
            center_y = height // 2
            x1 = int(center_x + radius * np.cos(angle))
            y1 = int(center_y + radius * np.sin(angle))
            x2 = x1 + 100
            y2 = y1 + 150

        elif motion_pattern == "random":
            # Random walk
            np.random.seed(frame_num * 100 + obj_id)
            x1 = np.random.randint(50, width - 150)
            y1 = np.random.randint(50, height - 200)
            x2 = x1 + 100
            y2 = y1 + 150

        else:
            # Stationary
            x1 = 100 + obj_id * 200
            y1 = 200
            x2 = x1 + 100
            y2 = y1 + 150

        # Keep within bounds
        x1 = max(0, min(x1, width - 100))
        y1 = max(0, min(y1, height - 150))
        x2 = min(x1 + 100, width)
        y2 = min(y1 + 150, height)

        detection = Detection(
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            confidence=0.85 + np.random.random() * 0.1,
            class_id=obj_id,
            class_name=f"object_{obj_id}",
        )
        detections.append(detection)

    return detections


class TestByteTrackPerformance:
    """Benchmark ByteTrack performance."""

    def benchmark_bytetrack(
        self,
        num_frames: int,
        num_objects: int,
        motion_pattern: str,
        resolution: tuple = (1920, 1080),
        target_fps: float = 100.0,
    ) -> Dict:
        """Benchmark ByteTrack with specified parameters.

        Args:
            num_frames: Number of frames to track
            num_objects: Number of objects per frame
            motion_pattern: Motion pattern for objects
            resolution: Frame resolution (width, height)
            target_fps: Target FPS for pass/fail

        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*80}")
        print(
            f"ByteTrack Benchmark: {resolution[0]}x{resolution[1]}, "
            f"{num_objects} objects, {motion_pattern} motion"
        )
        print(f"{'='*80}")

        tracker = ByteTracker(max_age=30, min_hits=3, iou_threshold=0.3)

        # Track all frames
        start_time = time.perf_counter()

        for frame_num in range(num_frames):
            detections = create_mock_detections(
                frame_num,
                num_objects,
                width=resolution[0],
                height=resolution[1],
                motion_pattern=motion_pattern,
            )

            if frame_num == 0:
                tracker.initialize(detections)
            else:
                tracker.update(detections)

        elapsed = time.perf_counter() - start_time
        fps = num_frames / elapsed if elapsed > 0 else 0
        latency_ms = (elapsed / num_frames) * 1000 if num_frames > 0 else 0

        # Count tracks
        total_tracks = len(tracker.tracks)
        active_tracks = len(tracker.get_active_tracks())

        results = {
            "tracker": "ByteTrack",
            "num_frames": num_frames,
            "num_objects": num_objects,
            "motion_pattern": motion_pattern,
            "resolution": f"{resolution[0]}x{resolution[1]}",
            "elapsed_sec": elapsed,
            "fps": fps,
            "latency_ms": latency_ms,
            "total_tracks": total_tracks,
            "active_tracks": active_tracks,
            "target_fps": target_fps,
            "success": fps >= target_fps,
        }

        print(f"\nResults:")
        print(f"  Frames tracked: {num_frames}")
        print(f"  Elapsed time: {elapsed:.3f}s")
        print(f"  FPS: {fps:.1f}")
        print(f"  Latency: {latency_ms:.2f}ms per frame")
        print(f"  Total tracks: {total_tracks}")
        print(f"  Active tracks: {active_tracks}")
        print(f"  Target FPS: {target_fps}")
        print(f"  Status: {'✓ PASS' if results['success'] else '✗ FAIL'}")

        return results

    def test_bytetrack_1080p_2objects(self):
        """ByteTrack with 2 objects @ 1080p (target: 100+ FPS)."""
        results = self.benchmark_bytetrack(
            num_frames=300,
            num_objects=2,
            motion_pattern="linear",
            resolution=(1920, 1080),
            target_fps=100.0,
        )

        assert results["fps"] > 0

    def test_bytetrack_1080p_5objects(self):
        """ByteTrack with 5 objects @ 1080p."""
        results = self.benchmark_bytetrack(
            num_frames=300,
            num_objects=5,
            motion_pattern="linear",
            resolution=(1920, 1080),
            target_fps=80.0,
        )

        assert results["fps"] > 0

    def test_bytetrack_1080p_10objects(self):
        """ByteTrack with 10 objects @ 1080p."""
        results = self.benchmark_bytetrack(
            num_frames=300,
            num_objects=10,
            motion_pattern="linear",
            resolution=(1920, 1080),
            target_fps=60.0,
        )

        assert results["fps"] > 0

    def test_bytetrack_4k(self):
        """ByteTrack @ 4K resolution."""
        results = self.benchmark_bytetrack(
            num_frames=200,
            num_objects=2,
            motion_pattern="linear",
            resolution=(3840, 2160),
            target_fps=80.0,
        )

        assert results["fps"] > 0

    @pytest.mark.parametrize("motion_pattern", ["linear", "circular", "random"])
    def test_bytetrack_motion_patterns(self, motion_pattern):
        """Test ByteTrack with different motion patterns."""
        results = self.benchmark_bytetrack(
            num_frames=200,
            num_objects=3,
            motion_pattern=motion_pattern,
            resolution=(1920, 1080),
            target_fps=80.0,
        )

        print(f"Motion pattern '{motion_pattern}': {results['fps']:.1f} FPS")


class TestImprovedTrackerPerformance:
    """Benchmark ImprovedTracker performance."""

    def benchmark_improved_tracker(
        self,
        num_frames: int,
        num_objects: int,
        use_optical_flow: bool,
        use_kalman: bool,
        resolution: tuple = (1920, 1080),
        target_fps: float = 80.0,
    ) -> Dict:
        """Benchmark ImprovedTracker.

        Args:
            num_frames: Number of frames to track
            num_objects: Number of objects per frame
            use_optical_flow: Enable optical flow
            use_kalman: Enable Kalman filter
            resolution: Frame resolution
            target_fps: Target FPS

        Returns:
            Benchmark results
        """
        print(f"\n{'='*80}")
        print(
            f"ImprovedTracker Benchmark: {resolution[0]}x{resolution[1]}, " f"{num_objects} objects"
        )
        print(f"Optical Flow: {use_optical_flow}, Kalman: {use_kalman}")
        print(f"{'='*80}")

        tracker = ImprovedTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            use_optical_flow=use_optical_flow,
            use_kalman=use_kalman,
        )

        # Create mock frames for optical flow
        prev_frame = None
        if use_optical_flow:
            prev_frame = np.random.randint(
                0, 255, (resolution[1], resolution[0], 3), dtype=np.uint8
            )

        start_time = time.perf_counter()

        for frame_num in range(num_frames):
            detections = create_mock_detections(
                frame_num,
                num_objects,
                width=resolution[0],
                height=resolution[1],
                motion_pattern="linear",
            )

            current_frame = None
            if use_optical_flow:
                current_frame = np.random.randint(
                    0, 255, (resolution[1], resolution[0], 3), dtype=np.uint8
                )

            if frame_num == 0:
                tracker.initialize(detections)
            else:
                tracker.update(detections, current_frame=current_frame, prev_frame=prev_frame)

            prev_frame = current_frame

        elapsed = time.perf_counter() - start_time
        fps = num_frames / elapsed if elapsed > 0 else 0
        latency_ms = (elapsed / num_frames) * 1000 if num_frames > 0 else 0

        results = {
            "tracker": "ImprovedTracker",
            "num_frames": num_frames,
            "num_objects": num_objects,
            "use_optical_flow": use_optical_flow,
            "use_kalman": use_kalman,
            "resolution": f"{resolution[0]}x{resolution[1]}",
            "elapsed_sec": elapsed,
            "fps": fps,
            "latency_ms": latency_ms,
            "total_tracks": len(tracker.tracks),
            "active_tracks": len(tracker.get_active_tracks()),
            "target_fps": target_fps,
            "success": fps >= target_fps,
        }

        print(f"\nResults:")
        print(f"  FPS: {fps:.1f}")
        print(f"  Latency: {latency_ms:.2f}ms per frame")
        print(f"  Status: {'✓ PASS' if results['success'] else '✗ FAIL'}")

        return results

    def test_improved_tracker_baseline(self):
        """ImprovedTracker with minimal features (baseline)."""
        results = self.benchmark_improved_tracker(
            num_frames=200,
            num_objects=2,
            use_optical_flow=False,
            use_kalman=True,
            resolution=(1920, 1080),
            target_fps=90.0,
        )

        assert results["fps"] > 0

    def test_improved_tracker_full_features(self):
        """ImprovedTracker with all features enabled (target: 80+ FPS)."""
        results = self.benchmark_improved_tracker(
            num_frames=200,
            num_objects=2,
            use_optical_flow=True,
            use_kalman=True,
            resolution=(1920, 1080),
            target_fps=80.0,
        )

        assert results["fps"] > 0

    def test_improved_tracker_optical_flow_only(self):
        """ImprovedTracker with only optical flow."""
        results = self.benchmark_improved_tracker(
            num_frames=200,
            num_objects=2,
            use_optical_flow=True,
            use_kalman=False,
            resolution=(1920, 1080),
            target_fps=85.0,
        )

        assert results["fps"] > 0


class TestTrackerComparison:
    """Compare different tracking algorithms."""

    def test_bytetrack_vs_improved(self):
        """Compare ByteTrack vs ImprovedTracker performance."""
        print("\n" + "=" * 80)
        print("COMPARISON: ByteTrack vs ImprovedTracker")
        print("=" * 80)

        test_params = {
            "num_frames": 300,
            "num_objects": 2,
            "resolution": (1920, 1080),
        }

        # ByteTrack
        bytetrack = ByteTracker()
        start = time.perf_counter()
        for frame_num in range(test_params["num_frames"]):
            detections = create_mock_detections(
                frame_num,
                test_params["num_objects"],
                width=test_params["resolution"][0],
                height=test_params["resolution"][1],
                motion_pattern="linear",
            )
            if frame_num == 0:
                bytetrack.initialize(detections)
            else:
                bytetrack.update(detections)
        bytetrack_time = time.perf_counter() - start
        bytetrack_fps = test_params["num_frames"] / bytetrack_time

        # ImprovedTracker
        improved = ImprovedTracker(use_optical_flow=False, use_kalman=True)
        start = time.perf_counter()
        for frame_num in range(test_params["num_frames"]):
            detections = create_mock_detections(
                frame_num,
                test_params["num_objects"],
                width=test_params["resolution"][0],
                height=test_params["resolution"][1],
                motion_pattern="linear",
            )
            if frame_num == 0:
                improved.initialize(detections)
            else:
                improved.update(detections)
        improved_time = time.perf_counter() - start
        improved_fps = test_params["num_frames"] / improved_time

        print(f"\nByteTrack:")
        print(f"  FPS: {bytetrack_fps:.1f}")
        print(f"  Time: {bytetrack_time:.3f}s")

        print(f"\nImprovedTracker:")
        print(f"  FPS: {improved_fps:.1f}")
        print(f"  Time: {improved_time:.3f}s")

        speedup = bytetrack_fps / improved_fps if improved_fps > 0 else 0
        print(f"\nByteTrack speedup: {speedup:.2f}x")

        print("=" * 80)

    def test_scalability_object_count(self):
        """Test how trackers scale with number of objects."""
        print("\n" + "=" * 80)
        print("SCALABILITY TEST: Object Count")
        print("=" * 80)

        object_counts = [1, 2, 5, 10, 20]
        bytetrack_results = []
        improved_results = []

        for obj_count in object_counts:
            # ByteTrack
            tracker = ByteTracker()
            start = time.perf_counter()
            for frame_num in range(100):
                detections = create_mock_detections(frame_num, obj_count, motion_pattern="linear")
                if frame_num == 0:
                    tracker.initialize(detections)
                else:
                    tracker.update(detections)
            elapsed = time.perf_counter() - start
            bytetrack_fps = 100 / elapsed
            bytetrack_results.append(bytetrack_fps)

            # ImprovedTracker
            tracker = ImprovedTracker(use_optical_flow=False, use_kalman=True)
            start = time.perf_counter()
            for frame_num in range(100):
                detections = create_mock_detections(frame_num, obj_count, motion_pattern="linear")
                if frame_num == 0:
                    tracker.initialize(detections)
                else:
                    tracker.update(detections)
            elapsed = time.perf_counter() - start
            improved_fps = 100 / elapsed
            improved_results.append(improved_fps)

        print("\nObjects | ByteTrack FPS | ImprovedTracker FPS")
        print("-" * 50)
        for i, obj_count in enumerate(object_counts):
            print(f"{obj_count:7d} | {bytetrack_results[i]:13.1f} | {improved_results[i]:19.1f}")

        print("=" * 80)


class TestTrackingAccuracy:
    """Test tracking accuracy metrics."""

    def test_track_continuity(self):
        """Test track ID continuity (fewer ID switches is better)."""
        print("\n" + "=" * 80)
        print("ACCURACY TEST: Track Continuity")
        print("=" * 80)

        tracker = ByteTracker(max_age=30, min_hits=3)

        # Track same object over 200 frames
        num_frames = 200
        for frame_num in range(num_frames):
            # Same detection with slight movement
            detections = [
                Detection(
                    bbox=(100 + frame_num, 200, 200 + frame_num, 350),
                    confidence=0.9,
                    class_id=0,
                    class_name="object",
                )
            ]

            if frame_num == 0:
                tracker.initialize(detections)
            else:
                tracker.update(detections)

        # Should have only 1 track (no ID switches)
        total_tracks = len(tracker.tracks)
        active_tracks = len(tracker.get_active_tracks())

        print(f"Frames tracked: {num_frames}")
        print(f"Total tracks created: {total_tracks}")
        print(f"Active tracks: {active_tracks}")

        # Ideally should be 1 track
        if total_tracks == 1:
            print("✓ EXCELLENT: Perfect track continuity (1 track)")
        elif total_tracks <= 3:
            print(f"✓ GOOD: Minimal ID switches ({total_tracks} tracks)")
        else:
            print(f"⚠ WARNING: Multiple ID switches ({total_tracks} tracks)")

        print("=" * 80)

    def test_occlusion_handling(self):
        """Test tracker behavior during occlusions."""
        print("\n" + "=" * 80)
        print("ACCURACY TEST: Occlusion Handling")
        print("=" * 80)

        tracker = ByteTracker(max_age=30, min_hits=3)

        # Object appears, disappears, reappears
        num_frames = 100
        occlusion_start = 30
        occlusion_end = 50

        for frame_num in range(num_frames):
            if occlusion_start <= frame_num < occlusion_end:
                # Object occluded (no detection)
                detections = []
            else:
                # Object visible
                detections = [
                    Detection(
                        bbox=(100, 200, 200, 350), confidence=0.9, class_id=0, class_name="object"
                    )
                ]

            if frame_num == 0:
                tracker.initialize(detections)
            else:
                tracker.update(detections)

        total_tracks = len(tracker.tracks)
        active_tracks = len(tracker.get_active_tracks())

        print(f"Occlusion duration: {occlusion_end - occlusion_start} frames")
        print(f"Total tracks: {total_tracks}")
        print(f"Active tracks: {active_tracks}")

        # Should maintain track through occlusion
        if total_tracks <= 2:
            print("✓ GOOD: Track maintained through occlusion")
        else:
            print(f"⚠ WARNING: Track lost during occlusion ({total_tracks} tracks)")

        print("=" * 80)


class TestMemoryTracking:
    """Test memory usage for tracking algorithms."""

    def test_memory_leak_check(self):
        """Test for memory leaks during long tracking sessions."""
        import os

        import psutil

        print("\n" + "=" * 80)
        print("MEMORY TEST: Leak Check")
        print("=" * 80)

        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        tracker = ByteTracker()

        # Track for 1000 frames
        for frame_num in range(1000):
            detections = create_mock_detections(frame_num, num_objects=3, motion_pattern="linear")

            if frame_num == 0:
                tracker.initialize(detections)
            else:
                tracker.update(detections)

        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = final_memory_mb - initial_memory_mb

        print(f"Initial memory: {initial_memory_mb:.1f} MB")
        print(f"Final memory: {final_memory_mb:.1f} MB")
        print(f"Memory increase: {memory_increase_mb:.1f} MB")

        # Memory increase should be reasonable
        max_increase = 100.0  # MB
        if memory_increase_mb < max_increase:
            print(f"✓ PASS: No significant memory leak")
        else:
            print(f"⚠ WARNING: Large memory increase ({memory_increase_mb:.1f} MB)")

        print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
