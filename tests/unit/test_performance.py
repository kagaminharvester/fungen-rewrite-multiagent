"""
Unit tests for performance.py

Tests performance monitoring, FPS tracking, and profiling utilities.
"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.performance import FrameMetrics, PerformanceMonitor, PerformanceStats, Profiler, profile


class TestFrameMetrics(unittest.TestCase):
    """Tests for FrameMetrics dataclass."""

    def test_frame_metrics_creation(self):
        """Test creating FrameMetrics object."""
        metrics = FrameMetrics(frame_number=1, timestamp=time.time(), processing_time_ms=20.0)

        self.assertEqual(metrics.frame_number, 1)
        self.assertGreater(metrics.timestamp, 0)
        self.assertEqual(metrics.processing_time_ms, 20.0)

    def test_frame_metrics_defaults(self):
        """Test FrameMetrics default values."""
        metrics = FrameMetrics(frame_number=1, timestamp=time.time(), processing_time_ms=20.0)

        # Check defaults
        self.assertEqual(metrics.decode_time_ms, 0.0)
        self.assertEqual(metrics.inference_time_ms, 0.0)
        self.assertEqual(metrics.tracking_time_ms, 0.0)
        self.assertEqual(metrics.postprocess_time_ms, 0.0)
        self.assertEqual(metrics.vram_used_gb, 0.0)


class TestPerformanceStats(unittest.TestCase):
    """Tests for PerformanceStats dataclass."""

    def test_performance_stats_defaults(self):
        """Test PerformanceStats default values."""
        stats = PerformanceStats()

        self.assertEqual(stats.total_frames, 0)
        self.assertEqual(stats.total_time_seconds, 0.0)
        self.assertEqual(stats.average_fps, 0.0)
        self.assertEqual(stats.current_fps, 0.0)


class TestPerformanceMonitor(unittest.TestCase):
    """Tests for PerformanceMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor(
            window_size=10, enable_profiling=True, log_interval=1000  # Disable logging during tests
        )

    def test_monitor_initialization(self):
        """Test monitor initializes correctly."""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(self.monitor.window_size, 10)
        self.assertTrue(self.monitor.enable_profiling)

    def test_frame_timing(self):
        """Test frame timing measurement."""
        self.monitor.start_frame(0)
        time.sleep(0.01)  # Simulate 10ms processing
        self.monitor.end_frame()

        # Should have recorded one frame
        self.assertEqual(len(self.monitor._frame_times), 1)

        # Frame time should be approximately 10ms
        frame_time = self.monitor._frame_times[0]
        self.assertGreater(frame_time, 0.009)
        self.assertLess(frame_time, 0.020)

    def test_multiple_frames(self):
        """Test processing multiple frames."""
        num_frames = 5

        for i in range(num_frames):
            self.monitor.start_frame(i)
            time.sleep(0.01)
            self.monitor.end_frame()

        # Should have recorded all frames
        self.assertEqual(len(self.monitor._frame_times), num_frames)
        self.assertEqual(self.monitor._total_frames, num_frames)

    def test_fps_calculation(self):
        """Test FPS calculation."""
        # Process frames at known rate
        for i in range(5):
            self.monitor.start_frame(i)
            time.sleep(0.02)  # 50 FPS target
            self.monitor.end_frame()

        fps = self.monitor.get_fps(rolling=True)

        # FPS should be approximately 50
        self.assertGreater(fps, 40)
        self.assertLess(fps, 60)

    def test_stage_profiling(self):
        """Test stage profiling."""
        self.monitor.start_frame(0)

        # Decode stage
        self.monitor.start_stage("decode")
        time.sleep(0.005)
        self.monitor.end_stage("decode")

        # Inference stage
        self.monitor.start_stage("inference")
        time.sleep(0.010)
        self.monitor.end_stage("inference")

        # Tracking stage
        self.monitor.start_stage("tracking")
        time.sleep(0.003)
        self.monitor.end_stage("tracking")

        # Postprocess stage
        self.monitor.start_stage("postprocess")
        time.sleep(0.002)
        self.monitor.end_stage("postprocess")

        self.monitor.end_frame()

        # Check that stages were recorded
        metrics = self.monitor._metrics[0]
        self.assertGreater(metrics.decode_time_ms, 4.0)
        self.assertGreater(metrics.inference_time_ms, 9.0)
        self.assertGreater(metrics.tracking_time_ms, 2.0)
        self.assertGreater(metrics.postprocess_time_ms, 1.0)

    def test_stats_aggregation(self):
        """Test statistics aggregation."""
        # Process multiple frames
        for i in range(10):
            self.monitor.start_frame(i)
            time.sleep(0.01)
            self.monitor.end_frame()

        stats = self.monitor.get_stats()

        # Check basic stats
        self.assertEqual(stats.total_frames, 10)
        self.assertGreater(stats.total_time_seconds, 0)
        self.assertGreater(stats.average_fps, 0)
        self.assertGreater(stats.current_fps, 0)
        self.assertGreater(stats.max_fps, 0)

    def test_window_size_limit(self):
        """Test rolling window size limit."""
        window_size = 5
        monitor = PerformanceMonitor(window_size=window_size, enable_profiling=False)

        # Process more frames than window size
        for i in range(10):
            monitor.start_frame(i)
            time.sleep(0.01)
            monitor.end_frame()

        # Frame times should be limited to window size
        self.assertEqual(len(monitor._frame_times), window_size)

    def test_reset(self):
        """Test monitor reset."""
        # Process some frames
        for i in range(5):
            self.monitor.start_frame(i)
            time.sleep(0.01)
            self.monitor.end_frame()

        # Reset monitor
        self.monitor.reset()

        # Should be cleared
        self.assertEqual(len(self.monitor._frame_times), 0)
        self.assertEqual(self.monitor._total_frames, 0)
        self.assertEqual(len(self.monitor._metrics), 0)

    def test_export_metrics(self):
        """Test exporting metrics to JSON."""
        # Process some frames
        for i in range(3):
            self.monitor.start_frame(i)
            self.monitor.start_stage("inference")
            time.sleep(0.01)
            self.monitor.end_stage("inference")
            self.monitor.end_frame()

        # Export to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            self.monitor.export_metrics(output_path)

            # Verify file exists
            self.assertTrue(output_path.exists())

            # Verify JSON structure
            with open(output_path) as f:
                data = json.load(f)

            self.assertIn("summary", data)
            self.assertIn("stage_breakdown", data)
            self.assertIn("detailed_metrics", data)

            # Check summary
            self.assertEqual(data["summary"]["total_frames"], 3)
            self.assertGreater(data["summary"]["average_fps"], 0)

            # Check detailed metrics
            self.assertEqual(len(data["detailed_metrics"]), 3)

    def test_vram_tracking(self):
        """Test VRAM tracking."""
        self.monitor.start_frame(0)
        time.sleep(0.01)
        self.monitor.end_frame()

        current, peak = self.monitor.get_vram_usage()

        # Should return non-negative values
        self.assertGreaterEqual(current, 0.0)
        self.assertGreaterEqual(peak, 0.0)


class TestProfiler(unittest.TestCase):
    """Tests for Profiler context manager."""

    def test_profiler_basic(self):
        """Test basic profiler usage."""
        with Profiler("test_operation") as prof:
            time.sleep(0.01)

        # Should have measured time
        self.assertGreater(prof.elapsed_ms, 9.0)
        self.assertLess(prof.elapsed_ms, 20.0)

    def test_profiler_with_monitor(self):
        """Test profiler integration with monitor."""
        monitor = PerformanceMonitor(enable_profiling=True, log_interval=1000)
        monitor.start_frame(0)

        with Profiler("decode", monitor):
            time.sleep(0.005)

        with Profiler("inference", monitor):
            time.sleep(0.010)

        monitor.end_frame()

        # Check that stages were recorded
        metrics = monitor._metrics[0]
        self.assertGreater(metrics.decode_time_ms, 4.0)
        self.assertGreater(metrics.inference_time_ms, 9.0)

    def test_profiler_nested(self):
        """Test nested profilers."""
        with Profiler("outer") as outer:
            time.sleep(0.01)
            with Profiler("inner") as inner:
                time.sleep(0.01)

        # Outer should be longer than inner
        self.assertGreater(outer.elapsed_ms, inner.elapsed_ms)


class TestProfileDecorator(unittest.TestCase):
    """Tests for profile decorator."""

    def test_profile_decorator(self):
        """Test profile decorator."""

        @profile("test_function")
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()
        self.assertEqual(result, 42)

    def test_profile_decorator_with_args(self):
        """Test profile decorator with function arguments."""

        @profile("add_function")
        def add(x, y):
            return x + y

        result = add(5, 3)
        self.assertEqual(result, 8)


class TestPerformanceMonitorThreadSafety(unittest.TestCase):
    """Tests for thread safety of PerformanceMonitor."""

    def test_concurrent_frame_updates(self):
        """Test concurrent frame updates (basic check)."""
        import threading

        monitor = PerformanceMonitor(window_size=100, enable_profiling=False, log_interval=1000)

        def process_frames(start_frame, count):
            for i in range(count):
                monitor.start_frame(start_frame + i)
                time.sleep(0.001)
                monitor.end_frame()

        # Create multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=process_frames, args=(i * 10, 10))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have processed all frames
        self.assertEqual(monitor._total_frames, 30)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_end_frame_without_start(self):
        """Test ending frame without starting."""
        monitor = PerformanceMonitor()
        monitor.end_frame()  # Should not crash

    def test_end_stage_without_start(self):
        """Test ending stage without starting."""
        monitor = PerformanceMonitor(enable_profiling=True)
        monitor.start_frame(0)
        monitor.end_stage("decode")  # Should not crash
        monitor.end_frame()

    def test_stats_without_frames(self):
        """Test getting stats without processing frames."""
        monitor = PerformanceMonitor()
        stats = monitor.get_stats()

        # Should return valid but empty stats
        self.assertEqual(stats.total_frames, 0)
        self.assertEqual(stats.average_fps, 0.0)

    def test_fps_with_zero_frames(self):
        """Test FPS calculation with zero frames."""
        monitor = PerformanceMonitor()
        fps = monitor.get_fps()

        # Should return 0, not crash
        self.assertEqual(fps, 0.0)


class TestPerformanceRegressionDetection(unittest.TestCase):
    """Tests for detecting performance regressions."""

    def test_consistent_timing(self):
        """Test that timing is consistent across multiple runs."""
        monitor = PerformanceMonitor(window_size=10)

        fps_samples = []

        for run in range(3):
            monitor.reset()

            # Process frames at consistent rate
            for i in range(10):
                monitor.start_frame(i)
                time.sleep(0.01)  # 100 FPS target
                monitor.end_frame()

            fps_samples.append(monitor.get_fps(rolling=True))

        # FPS should be consistent across runs (within 20%)
        avg_fps = sum(fps_samples) / len(fps_samples)
        for fps in fps_samples:
            self.assertGreater(fps, avg_fps * 0.8)
            self.assertLess(fps, avg_fps * 1.2)


if __name__ == "__main__":
    unittest.main()
