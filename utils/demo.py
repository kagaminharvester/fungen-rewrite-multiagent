#!/usr/bin/env python3
"""
Cross-Platform Utilities Demo

Demonstrates hardware detection, conditional imports, and performance monitoring
capabilities for the FunGen rewrite.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    PerformanceMonitor,
    Profiler,
    detect_hardware,
    get_device,
    get_performance_config,
    print_capabilities,
)


def demo_hardware_detection():
    """Demonstrate hardware detection."""
    print("\n" + "=" * 70)
    print("DEMO 1: Hardware Detection")
    print("=" * 70)

    hw_info = detect_hardware()

    print(f"\nDetected Platform: {hw_info.platform_profile.value}")
    print(f"Hardware Type: {hw_info.hardware_type.value}")
    print(f"Device: {hw_info.device_name}")

    if hw_info.device_count > 0:
        print(f"GPU Count: {hw_info.device_count}")
        print(f"Total VRAM: {hw_info.total_memory_gb:.2f} GB")
        print(f"Available VRAM: {hw_info.available_memory_gb:.2f} GB")

    print(f"\nCPU Architecture: {hw_info.cpu_info.get('architecture')}")
    print(f"CPU Cores: {hw_info.cpu_info.get('cpu_count')}")

    if hw_info.cpu_info.get("is_raspberry_pi"):
        print(f"Raspberry Pi: {hw_info.cpu_info.get('pi_model')}")


def demo_device_selection():
    """Demonstrate device selection."""
    print("\n" + "=" * 70)
    print("DEMO 2: PyTorch Device Selection")
    print("=" * 70)

    device_gpu = get_device(prefer_gpu=True)
    device_cpu = get_device(prefer_gpu=False)

    print(f"\nPreferred GPU device: {device_gpu}")
    print(f"CPU device: {device_cpu}")


def demo_performance_config():
    """Demonstrate performance configuration."""
    print("\n" + "=" * 70)
    print("DEMO 3: Performance Configuration")
    print("=" * 70)

    # 1080p configuration
    config_1080p = get_performance_config((1920, 1080))
    print("\n1080p Configuration:")
    print(f"  Batch Size: {config_1080p.batch_size}")
    print(f"  Workers: {config_1080p.num_workers}")
    print(f"  TensorRT: {config_1080p.use_tensorrt}")
    print(f"  FP16: {config_1080p.use_fp16}")
    print(f"  Optical Flow: {config_1080p.enable_optical_flow}")
    print(f"  ReID: {config_1080p.enable_reid}")
    print(f"  Target FPS: {config_1080p.target_fps}")
    print(f"  VRAM Limit: {config_1080p.vram_limit_gb:.1f} GB")

    # 8K configuration
    config_8k = get_performance_config((7680, 4320))
    print("\n8K Configuration:")
    print(f"  Batch Size: {config_8k.batch_size}")
    print(f"  Workers: {config_8k.num_workers}")
    print(f"  Target FPS: {config_8k.target_fps}")


def demo_capabilities():
    """Demonstrate capability detection."""
    print("\n" + "=" * 70)
    print("DEMO 4: GPU Capabilities")
    print("=" * 70)
    print()

    print_capabilities(detailed=True)


def demo_performance_monitoring():
    """Demonstrate performance monitoring."""
    print("\n" + "=" * 70)
    print("DEMO 5: Performance Monitoring")
    print("=" * 70)

    import time

    monitor = PerformanceMonitor(window_size=10, enable_profiling=True, log_interval=5)

    print("\nSimulating video processing...")

    # Simulate 20 frames
    for frame_num in range(20):
        monitor.start_frame(frame_num)

        # Simulate decode
        with Profiler("decode", monitor):
            time.sleep(0.002)

        # Simulate inference
        with Profiler("inference", monitor):
            time.sleep(0.008)

        # Simulate tracking
        with Profiler("tracking", monitor):
            time.sleep(0.003)

        # Simulate postprocess
        with Profiler("postprocess", monitor):
            time.sleep(0.001)

        monitor.end_frame()

    # Get final stats
    stats = monitor.get_stats()

    print(f"\n\nFinal Statistics:")
    print(f"  Total Frames: {stats.total_frames}")
    print(f"  Total Time: {stats.total_time_seconds:.2f}s")
    print(f"  Average FPS: {stats.average_fps:.2f}")
    print(f"  Current FPS: {stats.current_fps:.2f}")
    print(f"  Min FPS: {stats.min_fps:.2f}")
    print(f"  Max FPS: {stats.max_fps:.2f}")
    print(f"  Average Frame Time: {stats.average_frame_time_ms:.2f}ms")

    print(f"\n  Stage Breakdown:")
    print(f"    Decode: {stats.decode_time_percent:.1f}%")
    print(f"    Inference: {stats.inference_time_percent:.1f}%")
    print(f"    Tracking: {stats.tracking_time_percent:.1f}%")
    print(f"    Postprocess: {stats.postprocess_time_percent:.1f}%")

    # Export metrics
    output_path = Path("demo_performance_metrics.json")
    monitor.export_metrics(output_path)
    print(f"\n  Metrics exported to: {output_path}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("FunGen Cross-Platform Utilities Demo")
    print("=" * 70)

    demo_hardware_detection()
    demo_device_selection()
    demo_performance_config()
    demo_capabilities()
    demo_performance_monitoring()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- Seamless CPU/GPU detection and fallback")
    print("- Platform-specific configuration (Pi vs RTX 3090)")
    print("- Comprehensive performance monitoring")
    print("- Ready for 5+ FPS on Pi, 100+ FPS on RTX 3090")
    print("\nAll utilities are production-ready for the FunGen rewrite!")
    print()


if __name__ == "__main__":
    main()
