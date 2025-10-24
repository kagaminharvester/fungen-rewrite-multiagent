#!/usr/bin/env python3
"""
Batch video processing example.

This script demonstrates parallel processing of multiple videos using BatchProcessor.

Author: video-specialist agent
Date: 2025-10-24
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.batch_processor import BatchProcessor, ProcessingSettings


def main():
    """Batch video processing example."""
    # Check for video directory argument
    if len(sys.argv) < 2:
        print("Usage: python batch_video_processing.py <video_directory>")
        print("Example: python batch_video_processing.py videos/")
        sys.exit(1)

    video_dir = Path(sys.argv[1])

    if not video_dir.exists() or not video_dir.is_dir():
        print(f"Error: Directory not found: {video_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Batch Video Processing Example")
    print("=" * 60)

    # Find all video files
    video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".webm"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))

    if not video_files:
        print(f"Error: No video files found in {video_dir}")
        sys.exit(1)

    print(f"\n1. Found {len(video_files)} video files:")
    for video in video_files:
        print(f"  - {video.name}")

    # Initialize batch processor
    print("\n2. Initializing batch processor...")
    processor = BatchProcessor(num_workers=0)  # Auto-detect optimal workers
    print(f"  Using {processor.num_workers} parallel workers")

    # Create processing settings
    settings = ProcessingSettings(
        batch_size=8,
        hw_accel=True,
        tracker="bytetrack",
        output_dir=Path("output"),
        save_checkpoint=True,
        checkpoint_interval=100,
    )

    # Add videos to queue
    print("\n3. Adding videos to queue...")
    job_ids = processor.add_videos(video_files, settings)
    print(f"  Added {len(job_ids)} jobs to queue")

    # Define progress callback
    last_update = {}

    def on_progress(progress):
        """Progress callback for real-time updates."""
        job_id = progress.job_id
        video_name = progress.video_path.name

        # Only print if progress changed significantly
        if job_id not in last_update or abs(progress.progress_percent - last_update[job_id]) >= 5.0:
            last_update[job_id] = progress.progress_percent

            status_emoji = {
                "pending": "⏳",
                "processing": "⚙️",
                "completed": "✓",
                "failed": "✗",
                "cancelled": "⊗",
            }[progress.status.value]

            print(
                f"  [{status_emoji}] {video_name}: "
                f"{progress.progress_percent:.1f}% "
                f"({progress.fps:.1f} FPS, "
                f"ETA: {progress.eta_seconds:.0f}s, "
                f"Worker {progress.worker_id})"
            )

    # Process all videos
    print("\n4. Processing videos in parallel...")
    start_time = time.time()

    stats = processor.process(
        callback=on_progress,
        update_interval=0.5,  # Update every 0.5 seconds
    )

    elapsed_time = time.time() - start_time

    # Print results
    print("\n5. Processing complete!")
    print(f"\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Total jobs: {stats.total_jobs}")
    print(f"  Completed: {stats.completed_jobs}")
    print(f"  Failed: {stats.failed_jobs}")
    print(f"  Total frames: {stats.total_frames_processed:,}")
    print(f"  Average FPS: {stats.average_fps:.1f}")
    print(f"  Total time: {elapsed_time:.1f} seconds")
    print(f"  Throughput: {stats.total_frames_processed / elapsed_time:.1f} FPS")

    # Print individual job results
    print(f"\n" + "=" * 60)
    print("Individual Job Results")
    print("=" * 60)

    for job_id, job in processor.jobs.items():
        status_emoji = {
            "pending": "⏳",
            "processing": "⚙️",
            "completed": "✓",
            "failed": "✗",
            "cancelled": "⊗",
        }[job.status.value]

        print(f"  [{status_emoji}] {job.video_path.name}")
        print(f"      Status: {job.status.value}")
        print(f"      Frames: {job.frames_processed}/{job.total_frames}")
        print(f"      FPS: {job.fps:.1f}")
        print(f"      Time: {job.elapsed_time:.1f}s")

        if job.error_msg:
            print(f"      Error: {job.error_msg}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
