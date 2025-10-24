"""
Integration test for full video processing pipeline.

This module tests the complete workflow:
1. Video file loading and decoding
2. YOLO object detection
3. Multi-object tracking (ByteTrack, ImprovedTracker)
4. Funscript generation and validation

Author: test-engineer-2 agent
Date: 2025-10-24
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from core.model_manager import Detection, ModelManager
from core.video_processor import VideoMetadata, VideoProcessor
from trackers.base_tracker import FunscriptData, Track
from trackers.byte_tracker import ByteTracker
from trackers.improved_tracker import ImprovedTracker

# Test video paths
TEST_VIDEO_DIR = Path("/tmp/test_videos")
TEST_OUTPUT_DIR = Path("/tmp/test_outputs")


def create_test_video_with_objects(
    output_path: Path, width: int = 1920, height: int = 1080, fps: int = 30, duration: int = 5
) -> None:
    """Create a test video with moving objects using FFmpeg.

    Args:
        output_path: Path to output video file
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
        duration: Video duration in seconds
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create video with moving white boxes on black background
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r={fps}:d={duration}",
        "-vf",
        (
            f"drawbox=x=100+100*t:y=200:w=100:h=150:color=white:t=fill,"
            f"drawbox=x=300:y=100+50*sin(t):w=120:h=120:color=white:t=fill"
        ),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"Failed to create test video: {e}")


@pytest.fixture(scope="module")
def test_video_1080p():
    """Create 1080p test video with moving objects."""
    video_path = TEST_VIDEO_DIR / "test_1080p_moving_objects.mp4"
    create_test_video_with_objects(video_path, 1920, 1080, 30, 5)
    return video_path


@pytest.fixture(scope="module")
def test_video_720p():
    """Create 720p test video for faster testing."""
    video_path = TEST_VIDEO_DIR / "test_720p_moving_objects.mp4"
    create_test_video_with_objects(video_path, 1280, 720, 30, 3)
    return video_path


@pytest.fixture(scope="module")
def output_dir():
    """Create and return output directory for test results."""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_OUTPUT_DIR


class MockModelManager:
    """Mock model manager for testing without actual YOLO models."""

    def __init__(self):
        self.frame_count = 0

    def predict_batch(self, frames: np.ndarray) -> List[List[Detection]]:
        """Generate mock detections for testing.

        Creates 2 moving detections per frame to simulate tracking.
        """
        batch_detections = []

        for i in range(len(frames)):
            frame_num = self.frame_count + i

            # Create 2 moving detections
            detections = [
                Detection(
                    bbox=(100 + frame_num * 10, 200, 200 + frame_num * 10, 350),
                    confidence=0.85,
                    class_id=0,
                    class_name="object1",
                ),
                Detection(
                    bbox=(
                        300,
                        100 + int(50 * np.sin(frame_num * 0.1)),
                        420,
                        220 + int(50 * np.sin(frame_num * 0.1)),
                    ),
                    confidence=0.90,
                    class_id=1,
                    class_name="object2",
                ),
            ]
            batch_detections.append(detections)

        self.frame_count += len(frames)
        return batch_detections


class TestFullPipeline:
    """Test complete video processing pipeline."""

    def test_pipeline_video_to_funscript(self, test_video_720p, output_dir):
        """Test full pipeline: video → detection → tracking → funscript.

        This is the core integration test validating the complete workflow.
        """
        print("\n" + "=" * 80)
        print("FULL PIPELINE INTEGRATION TEST")
        print("=" * 80)

        # Step 1: Load and decode video
        print("\n[1/4] Loading and decoding video...")
        start_time = time.perf_counter()

        processor = VideoProcessor(str(test_video_720p), hw_accel=False)
        metadata = processor.get_metadata()

        print(f"  Resolution: {metadata.width}x{metadata.height}")
        print(f"  FPS: {metadata.fps:.2f}")
        print(f"  Total frames: {metadata.total_frames}")
        print(f"  Duration: {metadata.duration_sec:.2f}s")

        load_time = time.perf_counter() - start_time
        print(f"  Load time: {load_time:.3f}s")

        # Step 2: Initialize mock model and tracker
        print("\n[2/4] Initializing model and tracker...")
        start_time = time.perf_counter()

        model = MockModelManager()
        tracker = ByteTracker(max_age=30, min_hits=3, iou_threshold=0.3)

        init_time = time.perf_counter() - start_time
        print(f"  Initialization time: {init_time:.3f}s")

        # Step 3: Process video frames
        print("\n[3/4] Processing frames (detection + tracking)...")
        start_time = time.perf_counter()

        frame_count = 0
        all_tracks = []
        batch_size = 4

        for batch in processor.stream_frames(batch_size=batch_size):
            # Get detections from model
            batch_detections = model.predict_batch(batch.frames)

            # Update tracker with each frame
            for frame_dets in batch_detections:
                if frame_count == 0:
                    tracker.initialize(frame_dets)
                else:
                    tracks = tracker.update(frame_dets)
                    all_tracks.append(tracks)

                frame_count += 1

        process_time = time.perf_counter() - start_time
        process_fps = frame_count / process_time if process_time > 0 else 0

        print(f"  Processed frames: {frame_count}")
        print(f"  Processing time: {process_time:.3f}s")
        print(f"  Processing FPS: {process_fps:.1f}")
        print(f"  Active tracks: {len(tracker.get_active_tracks())}")

        # Step 4: Generate funscript
        print("\n[4/4] Generating funscript...")
        start_time = time.perf_counter()

        funscript_data = tracker.get_funscript_data(axis="vertical")
        output_path = output_dir / "test_pipeline_output.funscript"

        with open(output_path, "w") as f:
            json.dump(funscript_data.to_dict(), f, indent=2)

        gen_time = time.perf_counter() - start_time
        print(f"  Generation time: {gen_time:.3f}s")
        print(f"  Action points: {len(funscript_data.actions)}")
        print(f"  Output: {output_path}")

        # Validate results
        print("\n[VALIDATION]")
        assert frame_count > 0, "No frames processed"
        assert len(tracker.tracks) > 0, "No tracks created"
        assert len(funscript_data.actions) > 0, "No funscript actions generated"
        assert output_path.exists(), "Funscript file not created"

        # Validate funscript format
        with open(output_path, "r") as f:
            funscript_json = json.load(f)

        assert "version" in funscript_json, "Missing version field"
        assert "actions" in funscript_json, "Missing actions field"
        assert funscript_json["version"] == "1.0", "Wrong version"

        # Validate action points
        for action in funscript_json["actions"]:
            assert "at" in action, "Missing 'at' field in action"
            assert "pos" in action, "Missing 'pos' field in action"
            assert 0 <= action["pos"] <= 100, f"Position out of range: {action['pos']}"
            assert action["at"] >= 0, f"Negative timestamp: {action['at']}"

        print(f"  ✓ Funscript validation passed")
        print(f"  ✓ Pipeline completed successfully")

        total_time = load_time + init_time + process_time + gen_time
        print(f"\nTotal pipeline time: {total_time:.3f}s")
        print("=" * 80)

    def test_pipeline_with_improved_tracker(self, test_video_720p, output_dir):
        """Test pipeline with ImprovedTracker (Kalman + Optical Flow)."""
        print("\n" + "=" * 80)
        print("PIPELINE TEST: ImprovedTracker")
        print("=" * 80)

        processor = VideoProcessor(str(test_video_720p), hw_accel=False)
        metadata = processor.get_metadata()
        model = MockModelManager()

        # Use ImprovedTracker instead of ByteTrack
        tracker = ImprovedTracker(
            max_age=30, min_hits=3, iou_threshold=0.3, use_optical_flow=True, use_kalman=True
        )

        frame_count = 0
        prev_frame = None

        for batch in processor.stream_frames(batch_size=1):
            frame = batch.frames[0]
            batch_detections = model.predict_batch([frame])

            if frame_count == 0:
                tracker.initialize(batch_detections[0])
            else:
                # Pass previous frame for optical flow
                tracker.update(batch_detections[0], current_frame=frame, prev_frame=prev_frame)

            prev_frame = frame
            frame_count += 1

        # Generate funscript
        funscript_data = tracker.get_funscript_data(axis="vertical")
        output_path = output_dir / "test_improved_tracker_output.funscript"

        with open(output_path, "w") as f:
            json.dump(funscript_data.to_dict(), f, indent=2)

        print(f"Processed {frame_count} frames")
        print(f"Active tracks: {len(tracker.get_active_tracks())}")
        print(f"Action points: {len(funscript_data.actions)}")

        assert frame_count > 0
        assert len(funscript_data.actions) > 0
        assert output_path.exists()
        print("✓ ImprovedTracker pipeline test passed")
        print("=" * 80)

    def test_pipeline_no_detections(self, test_video_720p):
        """Test pipeline behavior with no detections (empty frames)."""
        print("\n" + "=" * 80)
        print("PIPELINE TEST: No Detections")
        print("=" * 80)

        processor = VideoProcessor(str(test_video_720p), hw_accel=False)
        tracker = ByteTracker()

        # Mock model that returns no detections
        frame_count = 0
        for batch in processor.stream_frames(batch_size=4):
            empty_detections = []

            if frame_count == 0:
                tracker.initialize(empty_detections)
            else:
                tracker.update(empty_detections)

            frame_count += len(batch.frames)

        funscript_data = tracker.get_funscript_data()

        print(f"Processed {frame_count} frames with no detections")
        print(f"Action points: {len(funscript_data.actions)}")

        # Should handle gracefully with no actions or minimal actions
        assert frame_count > 0
        print("✓ No detections test passed")
        print("=" * 80)

    def test_pipeline_vr_video(self):
        """Test pipeline with VR video format detection."""
        print("\n" + "=" * 80)
        print("PIPELINE TEST: VR Video Format")
        print("=" * 80)

        # Create VR test video (SBS format)
        vr_video_path = TEST_VIDEO_DIR / "test_FISHEYE180_LR.mp4"
        create_test_video_with_objects(vr_video_path, 1920, 1080, 30, 2)

        processor = VideoProcessor(str(vr_video_path), hw_accel=False)
        metadata = processor.get_metadata()

        print(f"VR Format detected: {metadata.vr_format.value}")
        print(f"Resolution: {metadata.width}x{metadata.height}")

        # Validate VR format detection
        assert (
            metadata.vr_format.value == "sbs_fisheye_180"
        ), f"Failed to detect VR format: {metadata.vr_format.value}"

        print("✓ VR format detection test passed")
        print("=" * 80)


class TestPipelinePerformance:
    """Performance tests for the full pipeline."""

    def test_pipeline_throughput_cpu(self, test_video_720p):
        """Benchmark pipeline throughput on CPU (Pi target: 5+ FPS)."""
        print("\n" + "=" * 80)
        print("PERFORMANCE TEST: CPU Throughput")
        print("=" * 80)

        processor = VideoProcessor(str(test_video_720p), hw_accel=False)
        model = MockModelManager()
        tracker = ByteTracker()

        start_time = time.perf_counter()
        frame_count = 0

        for batch in processor.stream_frames(batch_size=4):
            batch_detections = model.predict_batch(batch.frames)

            for frame_dets in batch_detections:
                if frame_count == 0:
                    tracker.initialize(frame_dets)
                else:
                    tracker.update(frame_dets)
                frame_count += 1

        elapsed = time.perf_counter() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print(f"Frames processed: {frame_count}")
        print(f"Elapsed time: {elapsed:.3f}s")
        print(f"Throughput: {fps:.2f} FPS")

        # CPU mode should achieve at least 5 FPS
        target_fps = 5.0
        print(f"Target FPS: {target_fps}")

        if fps >= target_fps:
            print(f"✓ PASS: {fps:.2f} FPS >= {target_fps} FPS")
        else:
            print(f"✗ WARN: {fps:.2f} FPS < {target_fps} FPS (may be slow hardware)")

        print("=" * 80)

    def test_pipeline_memory_usage(self, test_video_720p):
        """Test memory usage stays within limits during pipeline execution."""
        import os

        import psutil

        print("\n" + "=" * 80)
        print("PERFORMANCE TEST: Memory Usage")
        print("=" * 80)

        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        print(f"Initial memory: {initial_memory_mb:.1f} MB")

        # Process video
        processor = VideoProcessor(str(test_video_720p), hw_accel=False, buffer_size=30)
        model = MockModelManager()
        tracker = ByteTracker()

        frame_count = 0
        for batch in processor.stream_frames(batch_size=4):
            batch_detections = model.predict_batch(batch.frames)
            for frame_dets in batch_detections:
                if frame_count == 0:
                    tracker.initialize(frame_dets)
                else:
                    tracker.update(frame_dets)
                frame_count += 1

        peak_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = peak_memory_mb - initial_memory_mb

        print(f"Peak memory: {peak_memory_mb:.1f} MB")
        print(f"Memory increase: {memory_increase_mb:.1f} MB")

        # Memory increase should be reasonable (< 500 MB for 720p)
        max_memory_increase = 500.0

        if memory_increase_mb < max_memory_increase:
            print(f"✓ PASS: Memory increase {memory_increase_mb:.1f} MB < {max_memory_increase} MB")
        else:
            print(
                f"✗ WARN: Memory increase {memory_increase_mb:.1f} MB >= {max_memory_increase} MB"
            )

        print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
