"""
Performance profiling and monitoring utilities.

Provides real-time FPS tracking, VRAM monitoring, and performance profiling
for the FunGen rewrite. Supports both GPU and CPU-only environments.
"""

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from .conditional_imports import CUDA_AVAILABLE, TORCH_AVAILABLE, GPUMemoryManager, torch


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""

    frame_number: int
    timestamp: float
    processing_time_ms: float
    decode_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    vram_used_gb: float = 0.0


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""

    total_frames: int = 0
    total_time_seconds: float = 0.0
    average_fps: float = 0.0
    current_fps: float = 0.0
    min_fps: float = float("inf")
    max_fps: float = 0.0
    average_frame_time_ms: float = 0.0
    average_vram_gb: float = 0.0
    peak_vram_gb: float = 0.0
    decode_time_percent: float = 0.0
    inference_time_percent: float = 0.0
    tracking_time_percent: float = 0.0
    postprocess_time_percent: float = 0.0


class PerformanceMonitor:
    """Real-time performance monitoring for video processing."""

    def __init__(
        self, window_size: int = 30, enable_profiling: bool = True, log_interval: int = 100
    ):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of frames for rolling average (default: 30)
            enable_profiling: Enable detailed profiling (may add overhead)
            log_interval: Log stats every N frames
        """
        self.window_size = window_size
        self.enable_profiling = enable_profiling
        self.log_interval = log_interval

        # Frame timing
        self._frame_times: Deque[float] = deque(maxlen=window_size)
        self._frame_start_time: Optional[float] = None
        self._stage_start_time: Optional[float] = None

        # Detailed metrics
        self._metrics: List[FrameMetrics] = []
        self._current_frame: Optional[FrameMetrics] = None

        # Statistics
        self._total_frames = 0
        self._start_time = time.perf_counter()

        # VRAM tracking
        self._vram_samples: Deque[float] = deque(maxlen=window_size)
        self._peak_vram = 0.0

        # Thread safety
        self._lock = threading.Lock()

    def start_frame(self, frame_number: int) -> None:
        """
        Mark the start of frame processing.

        Args:
            frame_number: Current frame number
        """
        with self._lock:
            self._frame_start_time = time.perf_counter()
            self._current_frame = FrameMetrics(
                frame_number=frame_number, timestamp=self._frame_start_time, processing_time_ms=0.0
            )

    def end_frame(self) -> None:
        """Mark the end of frame processing and update statistics."""
        if self._frame_start_time is None:
            return

        with self._lock:
            frame_time = time.perf_counter() - self._frame_start_time
            self._frame_times.append(frame_time)
            self._total_frames += 1

            if self._current_frame is not None:
                self._current_frame.processing_time_ms = frame_time * 1000

                # Get VRAM usage
                if CUDA_AVAILABLE:
                    used, _ = GPUMemoryManager.get_memory_info()
                    self._current_frame.vram_used_gb = used
                    self._vram_samples.append(used)
                    self._peak_vram = max(self._peak_vram, used)

                if self.enable_profiling:
                    self._metrics.append(self._current_frame)

            # Log periodically
            if self._total_frames % self.log_interval == 0:
                self._log_stats()

            self._frame_start_time = None
            self._current_frame = None

    def start_stage(self, stage_name: str) -> None:
        """
        Mark the start of a processing stage.

        Args:
            stage_name: Name of stage ('decode', 'inference', 'tracking', 'postprocess')
        """
        if not self.enable_profiling or self._current_frame is None:
            return

        self._stage_start_time = time.perf_counter()

    def end_stage(self, stage_name: str) -> None:
        """
        Mark the end of a processing stage.

        Args:
            stage_name: Name of stage
        """
        if (
            not self.enable_profiling
            or self._stage_start_time is None
            or self._current_frame is None
        ):
            return

        with self._lock:
            stage_time = (time.perf_counter() - self._stage_start_time) * 1000

            if stage_name == "decode":
                self._current_frame.decode_time_ms = stage_time
            elif stage_name == "inference":
                self._current_frame.inference_time_ms = stage_time
            elif stage_name == "tracking":
                self._current_frame.tracking_time_ms = stage_time
            elif stage_name == "postprocess":
                self._current_frame.postprocess_time_ms = stage_time

            self._stage_start_time = None

    def get_fps(self, rolling: bool = True) -> float:
        """
        Get current FPS.

        Args:
            rolling: Use rolling window average (True) or overall average (False)

        Returns:
            Current FPS
        """
        with self._lock:
            if rolling:
                if len(self._frame_times) == 0:
                    return 0.0
                avg_time = sum(self._frame_times) / len(self._frame_times)
                return 1.0 / avg_time if avg_time > 0 else 0.0
            else:
                elapsed = time.perf_counter() - self._start_time
                return self._total_frames / elapsed if elapsed > 0 else 0.0

    def get_vram_usage(self) -> Tuple[float, float]:
        """
        Get VRAM usage statistics.

        Returns:
            Tuple of (current_vram_gb, peak_vram_gb)
        """
        if not CUDA_AVAILABLE:
            return (0.0, 0.0)

        with self._lock:
            current = self._vram_samples[-1] if self._vram_samples else 0.0
            return (current, self._peak_vram)

    def get_stats(self) -> PerformanceStats:
        """
        Get comprehensive performance statistics.

        Returns:
            PerformanceStats object
        """
        with self._lock:
            if self._total_frames == 0:
                return PerformanceStats()

            # Calculate FPS statistics
            current_fps = self.get_fps(rolling=True)
            average_fps = self.get_fps(rolling=False)

            # Calculate min/max FPS from frame times
            min_fps = float("inf")
            max_fps = 0.0
            if self._frame_times:
                for ft in self._frame_times:
                    fps = 1.0 / ft if ft > 0 else 0.0
                    min_fps = min(min_fps, fps)
                    max_fps = max(max_fps, fps)

            # Calculate average frame time
            avg_frame_time = (
                sum(self._frame_times) / len(self._frame_times) * 1000 if self._frame_times else 0.0
            )

            # Calculate VRAM statistics
            avg_vram = (
                sum(self._vram_samples) / len(self._vram_samples) if self._vram_samples else 0.0
            )

            # Calculate stage time percentages
            if self.enable_profiling and self._metrics:
                total_decode = sum(m.decode_time_ms for m in self._metrics)
                total_inference = sum(m.inference_time_ms for m in self._metrics)
                total_tracking = sum(m.tracking_time_ms for m in self._metrics)
                total_postprocess = sum(m.postprocess_time_ms for m in self._metrics)
                total_all = total_decode + total_inference + total_tracking + total_postprocess

                if total_all > 0:
                    decode_pct = (total_decode / total_all) * 100
                    inference_pct = (total_inference / total_all) * 100
                    tracking_pct = (total_tracking / total_all) * 100
                    postprocess_pct = (total_postprocess / total_all) * 100
                else:
                    decode_pct = inference_pct = tracking_pct = postprocess_pct = 0.0
            else:
                decode_pct = inference_pct = tracking_pct = postprocess_pct = 0.0

            total_time = time.perf_counter() - self._start_time

            return PerformanceStats(
                total_frames=self._total_frames,
                total_time_seconds=total_time,
                average_fps=average_fps,
                current_fps=current_fps,
                min_fps=min_fps if min_fps != float("inf") else 0.0,
                max_fps=max_fps,
                average_frame_time_ms=avg_frame_time,
                average_vram_gb=avg_vram,
                peak_vram_gb=self._peak_vram,
                decode_time_percent=decode_pct,
                inference_time_percent=inference_pct,
                tracking_time_percent=tracking_pct,
                postprocess_time_percent=postprocess_pct,
            )

    def _log_stats(self) -> None:
        """Log current statistics."""
        stats = self.get_stats()
        print(
            f"[Frame {self._total_frames}] "
            f"FPS: {stats.current_fps:.1f} "
            f"(avg: {stats.average_fps:.1f}, "
            f"min: {stats.min_fps:.1f}, "
            f"max: {stats.max_fps:.1f}) "
            f"| Frame time: {stats.average_frame_time_ms:.1f}ms"
        )

        if CUDA_AVAILABLE and self._vram_samples:
            print(
                f"         VRAM: {stats.average_vram_gb:.2f}GB "
                f"(peak: {stats.peak_vram_gb:.2f}GB)"
            )

        if self.enable_profiling:
            print(
                f"         Decode: {stats.decode_time_percent:.1f}% | "
                f"Inference: {stats.inference_time_percent:.1f}% | "
                f"Tracking: {stats.tracking_time_percent:.1f}% | "
                f"Post: {stats.postprocess_time_percent:.1f}%"
            )

    def export_metrics(self, output_path: Path) -> None:
        """
        Export detailed metrics to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        with self._lock:
            stats = self.get_stats()

            data = {
                "summary": {
                    "total_frames": stats.total_frames,
                    "total_time_seconds": stats.total_time_seconds,
                    "average_fps": stats.average_fps,
                    "current_fps": stats.current_fps,
                    "min_fps": stats.min_fps,
                    "max_fps": stats.max_fps,
                    "average_frame_time_ms": stats.average_frame_time_ms,
                    "average_vram_gb": stats.average_vram_gb,
                    "peak_vram_gb": stats.peak_vram_gb,
                },
                "stage_breakdown": {
                    "decode_percent": stats.decode_time_percent,
                    "inference_percent": stats.inference_time_percent,
                    "tracking_percent": stats.tracking_time_percent,
                    "postprocess_percent": stats.postprocess_time_percent,
                },
                "detailed_metrics": [],
            }

            if self.enable_profiling:
                for metric in self._metrics:
                    data["detailed_metrics"].append(
                        {
                            "frame": metric.frame_number,
                            "timestamp": metric.timestamp,
                            "processing_time_ms": metric.processing_time_ms,
                            "decode_time_ms": metric.decode_time_ms,
                            "inference_time_ms": metric.inference_time_ms,
                            "tracking_time_ms": metric.tracking_time_ms,
                            "postprocess_time_ms": metric.postprocess_time_ms,
                            "vram_used_gb": metric.vram_used_gb,
                        }
                    )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"Performance metrics exported to: {output_path}")

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._frame_times.clear()
            self._metrics.clear()
            self._vram_samples.clear()
            self._total_frames = 0
            self._start_time = time.perf_counter()
            self._peak_vram = 0.0
            self._frame_start_time = None
            self._stage_start_time = None
            self._current_frame = None


class Profiler:
    """Context manager for profiling code blocks."""

    def __init__(self, name: str, monitor: Optional[PerformanceMonitor] = None):
        """
        Initialize profiler.

        Args:
            name: Name of the code block being profiled
            monitor: Optional PerformanceMonitor to integrate with
        """
        self.name = name
        self.monitor = monitor
        self.start_time = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        """Start profiling."""
        self.start_time = time.perf_counter()
        if self.monitor:
            self.monitor.start_stage(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling."""
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        if self.monitor:
            self.monitor.end_stage(self.name)
        return False

    def print_time(self) -> None:
        """Print elapsed time."""
        print(f"{self.name}: {self.elapsed_ms:.2f}ms")


# Convenience function for quick profiling
def profile(name: str):
    """
    Decorator for profiling functions.

    Args:
        name: Name of the function being profiled

    Example:
        @profile("my_function")
        def my_function():
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with Profiler(name) as prof:
                result = func(*args, **kwargs)
            prof.print_time()
            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    # Demo/test mode
    print("FunGen Performance Monitor Demo\n")

    monitor = PerformanceMonitor(window_size=10, enable_profiling=True, log_interval=5)

    # Simulate frame processing
    for frame_num in range(20):
        monitor.start_frame(frame_num)

        # Simulate decode
        with Profiler("decode", monitor):
            time.sleep(0.005)

        # Simulate inference
        with Profiler("inference", monitor):
            time.sleep(0.010)

        # Simulate tracking
        with Profiler("tracking", monitor):
            time.sleep(0.003)

        # Simulate postprocess
        with Profiler("postprocess", monitor):
            time.sleep(0.002)

        monitor.end_frame()

    # Get final stats
    stats = monitor.get_stats()
    print(f"\nFinal Statistics:")
    print(f"Total Frames: {stats.total_frames}")
    print(f"Average FPS: {stats.average_fps:.2f}")
    print(f"Current FPS: {stats.current_fps:.2f}")
    print(f"Min FPS: {stats.min_fps:.2f}")
    print(f"Max FPS: {stats.max_fps:.2f}")
    print(f"Average Frame Time: {stats.average_frame_time_ms:.2f}ms")

    # Export metrics
    output_path = Path("performance_metrics.json")
    monitor.export_metrics(output_path)
