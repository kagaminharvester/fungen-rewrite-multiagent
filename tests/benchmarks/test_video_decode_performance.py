"""
Video decode performance benchmarks.

This module tests video decoding performance on different hardware configurations
and video resolutions. Target: 200+ FPS @ 1080p on RTX 3090.

Author: video-specialist agent
Date: 2025-10-24
"""

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from core.video_processor import HardwareAccel, VideoProcessor

# Mock video file for testing (will be created by conftest.py)
TEST_VIDEO_1080P = "/tmp/test_video_1080p.mp4"
TEST_VIDEO_4K = "/tmp/test_video_4k.mp4"
TEST_VIDEO_8K = "/tmp/test_video_8k.mp4"


def create_test_video(output_path: str, width: int, height: int, fps: int = 30, duration: int = 5):
    """Create a test video file using FFmpeg.

    Args:
        output_path: Path to output video
        width: Video width
        height: Video height
        fps: Frames per second
        duration: Duration in seconds
    """
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to create test video: {e}")


@pytest.fixture(scope="module")
def test_video_1080p():
    """Create 1080p test video."""
    if not Path(TEST_VIDEO_1080P).exists():
        create_test_video(TEST_VIDEO_1080P, 1920, 1080, fps=30, duration=5)
    return TEST_VIDEO_1080P


@pytest.fixture(scope="module")
def test_video_4k():
    """Create 4K test video."""
    if not Path(TEST_VIDEO_4K).exists():
        create_test_video(TEST_VIDEO_4K, 3840, 2160, fps=30, duration=5)
    return TEST_VIDEO_4K


@pytest.fixture(scope="module")
def test_video_8k():
    """Create 8K test video."""
    if not Path(TEST_VIDEO_8K).exists():
        create_test_video(TEST_VIDEO_8K, 7680, 4320, fps=30, duration=2)
    return TEST_VIDEO_8K


class TestVideoDecodePerformance:
    """Benchmark video decoding performance."""

    def benchmark_decode(
        self,
        video_path: str,
        hw_accel: bool,
        batch_size: int,
        target_fps: float,
    ) -> Dict:
        """Benchmark video decode performance.

        Args:
            video_path: Path to test video
            hw_accel: Enable hardware acceleration
            batch_size: Batch size for streaming
            target_fps: Target FPS for success

        Returns:
            Dict with benchmark results
        """
        processor = VideoProcessor(video_path, hw_accel=hw_accel)
        metadata = processor.get_metadata()

        print(f"\n{'='*60}")
        print(f"Benchmarking: {Path(video_path).name}")
        print(f"Resolution: {metadata.width}x{metadata.height}")
        print(f"FPS: {metadata.fps:.2f}")
        print(f"Codec: {metadata.codec}")
        print(f"Total frames: {metadata.total_frames}")
        print(f"HW Accel: {processor.hw_accel_type.value}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}")

        # Decode all frames
        start_time = time.perf_counter()
        total_frames = 0
        total_batches = 0

        for batch in processor.stream_frames(batch_size=batch_size):
            total_frames += batch.batch_size
            total_batches += 1

        elapsed = time.perf_counter() - start_time
        decode_fps = total_frames / elapsed if elapsed > 0 else 0

        # Get buffer stats
        buffer_stats = processor.get_buffer_stats()

        results = {
            "video_path": video_path,
            "resolution": f"{metadata.width}x{metadata.height}",
            "hw_accel": processor.hw_accel_type.value,
            "batch_size": batch_size,
            "total_frames": total_frames,
            "total_batches": total_batches,
            "elapsed_sec": elapsed,
            "decode_fps": decode_fps,
            "target_fps": target_fps,
            "success": decode_fps >= target_fps,
            "buffer_stats": buffer_stats,
        }

        print(f"\nResults:")
        print(f"  Total frames decoded: {total_frames}")
        print(f"  Total batches: {total_batches}")
        print(f"  Elapsed time: {elapsed:.3f}s")
        print(f"  Decode FPS: {decode_fps:.1f}")
        print(f"  Target FPS: {target_fps:.1f}")
        print(f"  Status: {'✓ PASS' if results['success'] else '✗ FAIL'}")
        print(f"  Buffer dropped frames: {buffer_stats['dropped_frames']}")
        print(f"  Buffer memory: {buffer_stats['memory_mb']:.1f} MB")

        return results

    def test_decode_1080p_gpu(self, test_video_1080p):
        """Benchmark 1080p GPU decode (RTX 3090 target: 200+ FPS)."""
        results = self.benchmark_decode(
            video_path=test_video_1080p,
            hw_accel=True,
            batch_size=8,
            target_fps=200.0,
        )

        # Assert target performance (may fail on non-GPU systems)
        if results["hw_accel"] == "nvdec":
            assert results["success"], f"GPU decode too slow: {results['decode_fps']:.1f} FPS"

    def test_decode_1080p_cpu(self, test_video_1080p):
        """Benchmark 1080p CPU decode (Pi target: 5+ FPS)."""
        results = self.benchmark_decode(
            video_path=test_video_1080p,
            hw_accel=False,
            batch_size=4,
            target_fps=5.0,
        )

        # CPU decode should achieve at least 5 FPS
        assert results["success"], f"CPU decode too slow: {results['decode_fps']:.1f} FPS"

    def test_decode_4k_gpu(self, test_video_4k):
        """Benchmark 4K GPU decode (RTX 3090 target: 100+ FPS)."""
        results = self.benchmark_decode(
            video_path=test_video_4k,
            hw_accel=True,
            batch_size=8,
            target_fps=100.0,
        )

        if results["hw_accel"] == "nvdec":
            assert results["success"], f"4K GPU decode too slow: {results['decode_fps']:.1f} FPS"

    def test_decode_8k_gpu(self, test_video_8k):
        """Benchmark 8K GPU decode (RTX 3090 target: 60+ FPS)."""
        results = self.benchmark_decode(
            video_path=test_video_8k,
            hw_accel=True,
            batch_size=8,
            target_fps=60.0,
        )

        if results["hw_accel"] == "nvdec":
            assert results["success"], f"8K GPU decode too slow: {results['decode_fps']:.1f} FPS"

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_batch_size_impact(self, test_video_1080p, batch_size):
        """Test impact of different batch sizes on performance.

        Larger batch sizes should improve throughput but may increase latency.
        """
        results = self.benchmark_decode(
            video_path=test_video_1080p,
            hw_accel=True,
            batch_size=batch_size,
            target_fps=50.0,  # Lower threshold for different batch sizes
        )

        # All batch sizes should achieve reasonable performance
        assert results["decode_fps"] > 10.0, f"Batch size {batch_size} too slow"

    def test_memory_usage(self, test_video_1080p):
        """Test that memory usage stays within limits."""
        processor = VideoProcessor(test_video_1080p, hw_accel=True, buffer_size=60)

        # Decode some frames
        frame_count = 0
        for batch in processor.stream_frames(batch_size=8):
            frame_count += batch.batch_size
            if frame_count >= 100:
                break

        buffer_stats = processor.get_buffer_stats()

        print(f"\nMemory usage after decoding {frame_count} frames:")
        print(f"  Buffer size: {buffer_stats['current_size']}/{buffer_stats['max_size']}")
        print(f"  Memory: {buffer_stats['memory_mb']:.1f} MB")
        print(f"  Dropped frames: {buffer_stats['dropped_frames']}")

        # For 1080p RGB frames: 1920*1080*3 bytes = ~6MB per frame
        # Buffer of 60 frames = ~360MB max
        max_expected_mb = 400.0
        assert (
            buffer_stats["memory_mb"] < max_expected_mb
        ), f"Memory usage too high: {buffer_stats['memory_mb']:.1f} MB"

    def test_vr_format_detection(self):
        """Test VR format detection from filenames."""
        test_cases = [
            ("video_FISHEYE180.mp4", "sbs_fisheye_180"),
            ("test_MKX200_4K.mp4", "sbs_fisheye_180"),
            ("video_LR_180.mp4", "sbs_fisheye_180"),
            ("video_TB_180.mp4", "tb_fisheye_180"),
            ("normal_video.mp4", "none"),
        ]

        for filename, expected_format in test_cases:
            # Create temporary test video
            temp_video = f"/tmp/{filename}"
            create_test_video(temp_video, 1920, 1080, fps=30, duration=1)

            processor = VideoProcessor(temp_video, hw_accel=False)
            metadata = processor.get_metadata()

            print(f"\nTesting: {filename}")
            print(f"  Detected format: {metadata.vr_format.value}")
            print(f"  Expected format: {expected_format}")

            assert (
                metadata.vr_format.value == expected_format
            ), f"Wrong VR format for {filename}: got {metadata.vr_format.value}, expected {expected_format}"


class TestVideoProcessorEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_video_path(self):
        """Test handling of invalid video path."""
        with pytest.raises(FileNotFoundError):
            VideoProcessor("/nonexistent/video.mp4")

    def test_seek_functionality(self, test_video_1080p):
        """Test seek to specific frame."""
        processor = VideoProcessor(test_video_1080p, hw_accel=False)
        metadata = processor.get_metadata()

        # Seek to middle of video
        target_frame = metadata.total_frames // 2
        processor.seek(target_frame)

        # Decode from that point
        batch = next(processor.stream_frames(batch_size=1))
        assert batch.metadata[0].frame_number == target_frame

    def test_seek_out_of_range(self, test_video_1080p):
        """Test seek with invalid frame number."""
        processor = VideoProcessor(test_video_1080p, hw_accel=False)
        metadata = processor.get_metadata()

        with pytest.raises(ValueError):
            processor.seek(metadata.total_frames + 100)

    def test_context_manager(self, test_video_1080p):
        """Test VideoProcessor as context manager."""
        with VideoProcessor(test_video_1080p, hw_accel=False) as processor:
            metadata = processor.get_metadata()
            assert metadata.width == 1920
            assert metadata.height == 1080

        # Processor should be cleaned up after context exit

    def test_partial_frame_range(self, test_video_1080p):
        """Test decoding only a portion of the video."""
        processor = VideoProcessor(test_video_1080p, hw_accel=False)
        metadata = processor.get_metadata()

        # Decode only frames 10-30
        start_frame = 10
        end_frame = 30
        decoded_frames = 0

        for batch in processor.stream_frames(
            batch_size=4,
            start_frame=start_frame,
            end_frame=end_frame,
        ):
            decoded_frames += batch.batch_size

        expected_frames = end_frame - start_frame
        assert (
            decoded_frames == expected_frames
        ), f"Expected {expected_frames} frames, got {decoded_frames}"


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "-s"])
