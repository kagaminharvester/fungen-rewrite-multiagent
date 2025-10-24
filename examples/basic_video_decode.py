#!/usr/bin/env python3
"""
Basic video decoding example.

This script demonstrates basic video decoding with the VideoProcessor class.

Author: video-specialist agent
Date: 2025-10-24
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.video_processor import VideoProcessor


def main():
    """Basic video decoding example."""
    # Check for video file argument
    if len(sys.argv) < 2:
        print("Usage: python basic_video_decode.py <video_file>")
        print("Example: python basic_video_decode.py video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    print("=" * 60)
    print("Basic Video Decoding Example")
    print("=" * 60)

    # Initialize video processor
    print(f"\n1. Loading video: {video_path}")
    processor = VideoProcessor(video_path, hw_accel=True)

    # Get metadata
    print("\n2. Getting video metadata...")
    metadata = processor.get_metadata()

    print(f"\nVideo Information:")
    print(f"  Resolution: {metadata.width}x{metadata.height}")
    print(f"  FPS: {metadata.fps:.2f}")
    print(f"  Duration: {metadata.duration_sec:.2f} seconds")
    print(f"  Total frames: {metadata.total_frames}")
    print(f"  Codec: {metadata.codec}")
    print(f"  Bitrate: {metadata.bitrate_mbps:.2f} Mbps")
    print(f"  VR format: {metadata.vr_format.value}")
    print(f"  HW Accel: {processor.hw_accel_type.value}")

    # Decode frames
    print("\n3. Decoding frames in batches...")
    batch_size = 8
    total_frames = 0
    batch_count = 0

    for batch in processor.stream_frames(batch_size=batch_size):
        total_frames += batch.batch_size
        batch_count += 1

        # Print progress every 10 batches
        if batch_count % 10 == 0:
            progress = (total_frames / metadata.total_frames) * 100
            print(f"  Progress: {total_frames}/{metadata.total_frames} frames ({progress:.1f}%)")

    print(f"\n4. Decoding complete!")
    print(f"  Total frames decoded: {total_frames}")
    print(f"  Total batches: {batch_count}")

    # Get buffer stats
    stats = processor.get_buffer_stats()
    print(f"\n5. Buffer statistics:")
    print(f"  Current size: {stats['current_size']}/{stats['max_size']} frames")
    print(f"  Memory usage: {stats['memory_mb']:.1f} MB")
    print(f"  Dropped frames: {stats['dropped_frames']}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
