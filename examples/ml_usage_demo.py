"""
ML Infrastructure Usage Demo

This script demonstrates how to use the ML infrastructure for
FunGen video processing with 100+ FPS on RTX 3090.

Author: ml-specialist agent
Date: 2025-10-24
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Config, ModelManager, TensorRTConverter, get_config, optimize_model_for_rtx3090


def demo_1_basic_inference():
    """Demo 1: Basic model loading and inference."""
    print("=" * 60)
    print("Demo 1: Basic Model Inference")
    print("=" * 60)

    # Auto-detect hardware configuration
    config = Config.auto_detect()
    print(f"Auto-detected profile: {config.name}")
    print(f"Device: {config.device}")
    print(f"TensorRT: {config.use_tensorrt}")
    print(f"FP16: {config.use_fp16}")
    print()

    # Initialize model manager
    manager = ModelManager(
        model_dir="models/", device=config.device, max_batch_size=config.max_batch_size
    )

    # Load model (will auto-detect best format)
    print("Loading YOLO model...")
    success = manager.load_model("yolo11n", optimize=True)

    if success:
        print(f"✓ Model loaded: {manager.model_info.format} ({manager.model_info.precision})")
        print(f"  VRAM usage: {manager.get_vram_usage():.2f} GB")
        print()

        # Create test frames
        print("Running inference on test frames...")
        frames = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(4)]

        # Run batch inference
        detections = manager.predict_batch(frames)

        print(f"✓ Processed {len(frames)} frames")
        print(f"  Detections per frame: {[len(d) for d in detections]}")

        # Print performance stats
        stats = manager.get_performance_stats()
        print(f"\nPerformance:")
        print(f"  FPS: {stats['avg_fps']:.1f}")
        print(f"  Latency: {stats['avg_latency_ms']:.1f} ms/frame")
        print(f"  VRAM: {stats['vram_usage_gb']:.2f} GB")
    else:
        print("✗ Model loading failed")


def demo_2_tensorrt_optimization():
    """Demo 2: TensorRT FP16 optimization."""
    print("\n" + "=" * 60)
    print("Demo 2: TensorRT Optimization")
    print("=" * 60)

    # Check if model exists
    model_path = Path("models/yolo11n.pt")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please download YOLO model first:")
        print("  from ultralytics import YOLO")
        print("  YOLO('yolo11n.pt').save('models/yolo11n.pt')")
        return

    # Optimize for RTX 3090
    print("Optimizing model for RTX 3090 (FP16)...")
    print("This may take 2-5 minutes...")

    engine_path = optimize_model_for_rtx3090(
        model_path=model_path, output_dir=Path("models/"), benchmark=True
    )

    if engine_path:
        print(f"\n✓ Optimization complete: {engine_path}")
    else:
        print("\n✗ Optimization failed (TensorRT may not be available)")


def demo_3_dynamic_batching():
    """Demo 3: Dynamic batch sizing based on VRAM."""
    print("\n" + "=" * 60)
    print("Demo 3: Dynamic Batch Sizing")
    print("=" * 60)

    config = Config.auto_detect()
    manager = ModelManager(
        model_dir="models/", device=config.device, max_batch_size=16  # Allow larger batches
    )

    success = manager.load_model("yolo11n", optimize=False)
    if not success:
        print("Model loading failed")
        return

    print(f"Model loaded: {manager.model_info.format}")
    print(f"Current VRAM: {manager.get_vram_usage():.2f} GB")
    print()

    # Test different available VRAM scenarios
    test_scenarios = [
        (24.0, "Full RTX 3090 (24GB)"),
        (10.0, "Half VRAM available"),
        (5.0, "Limited VRAM"),
        (2.0, "Minimal VRAM"),
    ]

    print("Optimal batch sizes for different VRAM scenarios:")
    for available_vram, description in test_scenarios:
        optimal_batch = manager.get_optimal_batch_size(available_vram)
        print(f"  {description:30s} → batch size: {optimal_batch}")


def demo_4_resolution_optimization():
    """Demo 4: Resolution-specific optimization."""
    print("\n" + "=" * 60)
    print("Demo 4: Resolution-Specific Optimization")
    print("=" * 60)

    config = Config.auto_detect()

    resolutions = [
        (1920, 1080, "1080p"),
        (2560, 1440, "1440p"),
        (3840, 2160, "4K"),
        (7680, 4320, "8K"),
    ]

    print("Optimal settings for different resolutions:")
    print(f"{'Resolution':15s} {'Batch':8s} {'Resize':8s} {'Target FPS':12s}")
    print("-" * 50)

    for width, height, name in resolutions:
        settings = config.get_optimal_settings_for_resolution(width, height)
        print(
            f"{name:15s} "
            f"{settings['batch_size']:8d} "
            f"{settings['resize_factor']:8.2f} "
            f"{settings['target_fps']:12d}"
        )


def demo_5_video_processing():
    """Demo 5: Real video processing pipeline."""
    print("\n" + "=" * 60)
    print("Demo 5: Video Processing Pipeline")
    print("=" * 60)

    # Check if test video exists
    video_path = Path("test_video.mp4")
    if not video_path.exists():
        print(f"Test video not found: {video_path}")
        print("Skipping video processing demo")
        return

    config = Config.auto_detect()
    manager = ModelManager(
        model_dir="models/", device=config.device, max_batch_size=config.max_batch_size
    )

    success = manager.load_model("yolo11n", optimize=True)
    if not success:
        print("Model loading failed")
        return

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Failed to open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    print()

    # Get optimal settings
    settings = config.get_optimal_settings_for_resolution(width, height)
    batch_size = settings["batch_size"]

    print(f"Processing with batch size: {batch_size}")
    print("Processing frames...")

    frame_count = 0
    batch = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        batch.append(frame)
        frame_count += 1

        # Process batch when full
        if len(batch) >= batch_size:
            detections = manager.predict_batch(batch)

            # Count total detections
            total_dets = sum(len(d) for d in detections)
            print(f"  Frames {frame_count-len(batch)+1}-{frame_count}: {total_dets} detections")

            batch.clear()

        # Process first 100 frames for demo
        if frame_count >= 100:
            break

    cap.release()

    # Print final stats
    stats = manager.get_performance_stats()
    print(f"\nFinal Performance:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Average FPS: {stats['avg_fps']:.1f}")
    print(f"  Latency: {stats['avg_latency_ms']:.1f} ms/frame")
    print(f"  VRAM usage: {stats['vram_usage_gb']:.2f} GB")
    print(f"  VRAM peak: {stats['vram_peak_gb']:.2f} GB")


def demo_6_config_management():
    """Demo 6: Configuration management."""
    print("\n" + "=" * 60)
    print("Demo 6: Configuration Management")
    print("=" * 60)

    # Show available profiles
    from core.config import PROFILES

    print("Available hardware profiles:")
    for name, profile in PROFILES.items():
        print(f"\n{name}:")
        print(f"  Device: {profile.device}")
        print(f"  TensorRT: {profile.use_tensorrt}")
        print(f"  FP16: {profile.use_fp16}")
        print(f"  Batch size: {profile.max_batch_size}")
        print(f"  Optical flow: {profile.enable_optical_flow}")
        print(f"  ReID: {profile.enable_reid}")

    # Auto-detect
    print("\nAuto-detected configuration:")
    config = Config.auto_detect()
    print(config)

    # Save config
    config_path = Path("config.json")
    config.save(config_path)
    print(f"\n✓ Configuration saved to {config_path}")

    # Load config
    loaded_config = Config.load(config_path)
    print(f"✓ Configuration loaded from {config_path}")
    print(f"  Profile: {loaded_config.name}")


def main():
    """Run all demos."""
    print("FunGen ML Infrastructure Demo")
    print("Author: ml-specialist agent")
    print()

    try:
        # Run demos
        demo_1_basic_inference()
        # demo_2_tensorrt_optimization()  # Commented out - takes 2-5 minutes
        demo_3_dynamic_batching()
        demo_4_resolution_optimization()
        # demo_5_video_processing()  # Commented out - needs test video
        demo_6_config_management()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
