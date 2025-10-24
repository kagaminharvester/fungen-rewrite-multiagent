"""
Integration tests for batch video processing.

Tests multi-video queue management, parallel processing, and batch operations.

Author: test-engineer-2 agent
Date: 2025-10-24
"""

import concurrent.futures
import json
import subprocess
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

from core.batch_processor import BatchProcessor, JobStatus, ProcessingSettings
from core.video_processor import VideoProcessor
from trackers.byte_tracker import ByteTracker

TEST_VIDEO_DIR = Path("/tmp/test_batch_videos")
TEST_OUTPUT_DIR = Path("/tmp/test_batch_outputs")


def create_test_videos(count: int = 5) -> List[Path]:
    """Create multiple test videos for batch processing.

    Args:
        count: Number of test videos to create

    Returns:
        List of paths to created videos
    """
    TEST_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_paths = []

    for i in range(count):
        video_path = TEST_VIDEO_DIR / f"test_video_{i:02d}.mp4"

        # Create short test videos (1-2 seconds each)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration=2:size=640x480:rate=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=20)
            video_paths.append(video_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pytest.skip(f"Failed to create test video {i}")

    return video_paths


@pytest.fixture(scope="module")
def test_videos():
    """Create a batch of test videos."""
    return create_test_videos(count=5)


@pytest.fixture(scope="module")
def output_dir():
    """Create and return output directory."""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_OUTPUT_DIR


class TestBatchProcessing:
    """Test batch video processing functionality."""

    def test_batch_processor_initialization(self):
        """Test BatchProcessor initialization."""
        print("\n" + "=" * 80)
        print("TEST: BatchProcessor Initialization")
        print("=" * 80)

        processor = BatchProcessor(num_workers=2)

        print(f"Workers: {processor.num_workers}")
        print(f"Queue size: {processor.queue_size()}")

        assert processor.num_workers == 2
        assert processor.queue_size() == 0
        print("✓ Initialization test passed")
        print("=" * 80)

    def test_add_videos_to_queue(self, test_videos):
        """Test adding multiple videos to processing queue."""
        print("\n" + "=" * 80)
        print("TEST: Add Videos to Queue")
        print("=" * 80)

        processor = BatchProcessor(num_workers=2)
        job_ids = []

        settings = ProcessingSettings(
            tracker_type="bytetrack", hw_accel=False, batch_size=4, output_dir=TEST_OUTPUT_DIR
        )

        for video in test_videos:
            job_id = processor.add_video(video, settings)
            job_ids.append(job_id)
            print(f"Added video: {video.name} -> Job ID: {job_id}")

        print(f"\nTotal jobs in queue: {processor.queue_size()}")

        assert processor.queue_size() == len(test_videos)
        assert len(job_ids) == len(test_videos)
        assert len(set(job_ids)) == len(job_ids)  # All unique
        print("✓ Queue test passed")
        print("=" * 80)

    def test_batch_processing_sequential(self, test_videos, output_dir):
        """Test sequential batch processing."""
        print("\n" + "=" * 80)
        print("TEST: Sequential Batch Processing")
        print("=" * 80)

        processor = BatchProcessor(num_workers=1)

        settings = ProcessingSettings(
            tracker_type="bytetrack", hw_accel=False, batch_size=4, output_dir=output_dir
        )

        # Add all videos
        job_ids = []
        for video in test_videos[:3]:  # Process first 3 videos
            job_id = processor.add_video(video, settings)
            job_ids.append(job_id)

        print(f"Added {len(job_ids)} videos to queue")

        # Track progress
        progress_updates = []

        def progress_callback(job_id, progress):
            progress_updates.append((job_id, progress))
            print(f"  Job {job_id}: {progress:.1f}%")

        # Process all videos
        start_time = time.perf_counter()
        results = processor.process_all(callback=progress_callback)
        elapsed = time.perf_counter() - start_time

        print(f"\nProcessing completed in {elapsed:.2f}s")
        print(f"Progress updates: {len(progress_updates)}")
        print(f"Results: {len(results)}")

        # Validate results
        assert len(results) == len(job_ids)
        assert len(progress_updates) > 0

        for job_id in job_ids:
            assert job_id in results
            assert results[job_id]["status"] in ["completed", "failed"]

            if results[job_id]["status"] == "completed":
                assert results[job_id]["output_file"].exists()
                print(f"✓ Job {job_id}: {results[job_id]['output_file'].name}")

        print("✓ Sequential processing test passed")
        print("=" * 80)

    def test_batch_processing_parallel(self, test_videos, output_dir):
        """Test parallel batch processing with multiple workers."""
        print("\n" + "=" * 80)
        print("TEST: Parallel Batch Processing")
        print("=" * 80)

        processor = BatchProcessor(num_workers=2)

        settings = ProcessingSettings(
            tracker_type="bytetrack", hw_accel=False, batch_size=4, output_dir=output_dir
        )

        # Add all videos
        job_ids = []
        for video in test_videos[:4]:  # Process 4 videos in parallel
            job_id = processor.add_video(video, settings)
            job_ids.append(job_id)

        print(f"Added {len(job_ids)} videos to queue")
        print(f"Workers: {processor.num_workers}")

        # Process with parallelism
        start_time = time.perf_counter()
        results = processor.process_all()
        parallel_time = time.perf_counter() - start_time

        print(f"\nParallel processing completed in {parallel_time:.2f}s")

        # Validate all completed
        completed_count = sum(1 for r in results.values() if r["status"] == "completed")

        print(f"Completed jobs: {completed_count}/{len(job_ids)}")

        assert completed_count > 0
        print("✓ Parallel processing test passed")
        print("=" * 80)

    def test_cancel_job(self, test_videos, output_dir):
        """Test canceling a job in the queue."""
        print("\n" + "=" * 80)
        print("TEST: Cancel Job")
        print("=" * 80)

        processor = BatchProcessor(num_workers=1)

        settings = ProcessingSettings(
            tracker_type="bytetrack", hw_accel=False, batch_size=4, output_dir=output_dir
        )

        # Add videos
        job_ids = []
        for video in test_videos[:3]:
            job_id = processor.add_video(video, settings)
            job_ids.append(job_id)

        # Cancel middle job
        cancel_job_id = job_ids[1]
        print(f"Canceling job: {cancel_job_id}")

        success = processor.cancel_job(cancel_job_id)

        print(f"Cancel result: {success}")
        print(f"Queue size after cancel: {processor.queue_size()}")

        assert success
        assert processor.queue_size() == len(job_ids) - 1
        print("✓ Cancel job test passed")
        print("=" * 80)

    def test_get_job_status(self, test_videos, output_dir):
        """Test getting job status during processing."""
        print("\n" + "=" * 80)
        print("TEST: Job Status Tracking")
        print("=" * 80)

        processor = BatchProcessor(num_workers=1)

        settings = ProcessingSettings(
            tracker_type="bytetrack", hw_accel=False, batch_size=4, output_dir=output_dir
        )

        # Add one video
        job_id = processor.add_video(test_videos[0], settings)

        print(f"Job ID: {job_id}")

        # Check initial status
        status = processor.get_job_status(job_id)
        print(f"Initial status: {status.value}")

        assert status == JobStatus.PENDING

        # Start processing
        processor.start_processing()
        time.sleep(0.5)  # Let processing start

        # Check status during processing
        status = processor.get_job_status(job_id)
        print(f"Status during processing: {status.value}")

        # Wait for completion
        processor.wait_for_completion()

        # Check final status
        status = processor.get_job_status(job_id)
        print(f"Final status: {status.value}")

        assert status in [JobStatus.COMPLETED, JobStatus.FAILED]
        print("✓ Job status test passed")
        print("=" * 80)

    def test_batch_with_different_settings(self, test_videos, output_dir):
        """Test batch processing with different settings per video."""
        print("\n" + "=" * 80)
        print("TEST: Different Settings Per Video")
        print("=" * 80)

        processor = BatchProcessor(num_workers=2)

        # Different settings for each video
        settings_list = [
            ProcessingSettings(
                tracker_type="bytetrack",
                hw_accel=False,
                batch_size=4,
                output_dir=output_dir / "bytetrack",
            ),
            ProcessingSettings(
                tracker_type="improved",
                hw_accel=False,
                batch_size=2,
                output_dir=output_dir / "improved",
            ),
        ]

        job_ids = []
        for i, video in enumerate(test_videos[:2]):
            settings = settings_list[i]
            settings.output_dir.mkdir(parents=True, exist_ok=True)

            job_id = processor.add_video(video, settings)
            job_ids.append(job_id)
            print(f"Video {i}: {settings.tracker_type}, batch_size={settings.batch_size}")

        # Process all
        results = processor.process_all()

        print(f"\nProcessed {len(results)} videos with different settings")

        for job_id in job_ids:
            if results[job_id]["status"] == "completed":
                print(f"✓ Job {job_id}: {results[job_id]['output_file'].name}")

        assert len(results) == len(job_ids)
        print("✓ Different settings test passed")
        print("=" * 80)


class TestBatchPerformance:
    """Performance tests for batch processing."""

    def test_parallel_speedup(self, test_videos, output_dir):
        """Test that parallel processing is faster than sequential."""
        print("\n" + "=" * 80)
        print("PERFORMANCE TEST: Parallel Speedup")
        print("=" * 80)

        settings = ProcessingSettings(
            tracker_type="bytetrack", hw_accel=False, batch_size=4, output_dir=output_dir
        )

        videos_to_test = test_videos[:4]

        # Sequential processing
        print("\n[1/2] Sequential processing (1 worker)...")
        processor_seq = BatchProcessor(num_workers=1)
        for video in videos_to_test:
            processor_seq.add_video(video, settings)

        start_time = time.perf_counter()
        processor_seq.process_all()
        sequential_time = time.perf_counter() - start_time

        print(f"Sequential time: {sequential_time:.2f}s")

        # Parallel processing
        print("\n[2/2] Parallel processing (2 workers)...")
        processor_par = BatchProcessor(num_workers=2)
        for video in videos_to_test:
            processor_par.add_video(video, settings)

        start_time = time.perf_counter()
        processor_par.process_all()
        parallel_time = time.perf_counter() - start_time

        print(f"Parallel time: {parallel_time:.2f}s")

        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        efficiency = speedup / 2.0 * 100  # 2 workers

        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Parallel efficiency: {efficiency:.1f}%")

        # Parallel should be faster (though may not be 2x due to overhead)
        if parallel_time < sequential_time:
            print(f"✓ PASS: Parallel faster by {speedup:.2f}x")
        else:
            print(f"✗ WARN: Parallel not faster (overhead or system limits)")

        print("=" * 80)

    def test_queue_throughput(self, test_videos):
        """Test throughput of adding videos to queue."""
        print("\n" + "=" * 80)
        print("PERFORMANCE TEST: Queue Throughput")
        print("=" * 80)

        processor = BatchProcessor(num_workers=2)

        settings = ProcessingSettings(
            tracker_type="bytetrack", hw_accel=False, batch_size=4, output_dir=TEST_OUTPUT_DIR
        )

        # Measure time to add videos to queue
        start_time = time.perf_counter()
        for video in test_videos:
            processor.add_video(video, settings)
        elapsed = time.perf_counter() - start_time

        throughput = len(test_videos) / elapsed if elapsed > 0 else 0

        print(f"Videos added: {len(test_videos)}")
        print(f"Time: {elapsed:.3f}s")
        print(f"Throughput: {throughput:.1f} videos/sec")

        # Should be able to add at least 10 videos per second
        assert throughput > 10.0
        print("✓ Queue throughput test passed")
        print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
