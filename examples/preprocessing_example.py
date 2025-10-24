#!/usr/bin/env python3
"""
Frame preprocessing example.

This script demonstrates frame preprocessing including resizing, cropping,
normalization, and VR video handling.

Author: video-specialist agent
Date: 2025-10-24
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.preprocessing import (
    FrameAnalyzer,
    FramePreprocessor,
    PreprocessConfig,
    VRPreprocessor,
    create_preprocessing_config_for_model,
)
from core.video_processor import VideoProcessor, VRFormat


def main():
    """Preprocessing example."""
    if len(sys.argv) < 2:
        print("Usage: python preprocessing_example.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]

    print("=" * 60)
    print("Frame Preprocessing Example")
    print("=" * 60)

    # Initialize video processor
    print(f"\n1. Loading video: {video_path}")
    processor = VideoProcessor(video_path, hw_accel=True)
    metadata = processor.get_metadata()

    print(f"\nVideo: {metadata.width}x{metadata.height} @ {metadata.fps:.2f} FPS")
    print(f"VR format: {metadata.vr_format.value}")

    # Example 1: Basic preprocessing
    print("\n" + "=" * 60)
    print("Example 1: Basic Preprocessing")
    print("=" * 60)

    config = PreprocessConfig(
        target_size=(640, 640),
        normalize=True,
        maintain_aspect=True,
    )

    preprocessor = FramePreprocessor(config)

    # Get first frame
    batch = next(processor.stream_frames(batch_size=1))
    frame = batch.frames[0]

    print(f"\nOriginal frame shape: {frame.shape}")
    print(f"Original dtype: {frame.dtype}")

    processed = preprocessor.process_frame(frame)

    print(f"Processed frame shape: {processed.shape}")
    print(f"Processed dtype: {processed.dtype}")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")

    # Example 2: YOLO preprocessing
    print("\n" + "=" * 60)
    print("Example 2: YOLO Model Preprocessing")
    print("=" * 60)

    yolo_config = create_preprocessing_config_for_model("yolo11n", input_size=640)
    yolo_preprocessor = FramePreprocessor(yolo_config)

    processed_yolo = yolo_preprocessor.process_frame(frame)
    print(f"\nYOLO preprocessed shape: {processed_yolo.shape}")
    print(f"Maintains aspect ratio: {yolo_config.maintain_aspect}")
    print(f"Normalization: {yolo_config.normalize}")

    # Example 3: VR video processing
    print("\n" + "=" * 60)
    print("Example 3: VR Video Processing")
    print("=" * 60)

    if metadata.vr_format != VRFormat.NONE:
        print(f"\nDetected VR format: {metadata.vr_format.value}")

        if "sbs" in metadata.vr_format.value:
            print("Splitting side-by-side frame...")
            left, right = VRPreprocessor.split_sbs(frame)
            print(f"  Left eye: {left.shape}")
            print(f"  Right eye: {right.shape}")

            # Process left eye only (common for tracking)
            processed_left = preprocessor.process_frame(left)
            print(f"  Processed left eye: {processed_left.shape}")

        elif "tb" in metadata.vr_format.value:
            print("Splitting top-bottom frame...")
            left, right = VRPreprocessor.split_tb(frame)
            print(f"  Left eye: {left.shape}")
            print(f"  Right eye: {right.shape}")

        if "fisheye" in metadata.vr_format.value:
            print("\nUndistorting fisheye...")
            # Note: This is computationally expensive
            undistorted = VRPreprocessor.undistort_fisheye(left, fov_degrees=180)
            print(f"  Undistorted shape: {undistorted.shape}")
    else:
        print("\nNot a VR video, skipping VR processing")

    # Example 4: Frame analysis
    print("\n" + "=" * 60)
    print("Example 4: Frame Analysis")
    print("=" * 60)

    analyzer = FrameAnalyzer()

    # Analyze first frame
    brightness = analyzer.calculate_brightness(frame)
    contrast = analyzer.calculate_contrast(frame)
    blur_score = analyzer.detect_blur(frame)

    print(f"\nFrame 0 analysis:")
    print(f"  Brightness: {brightness:.1f}/255")
    print(f"  Contrast: {contrast:.1f}")
    print(f"  Blur score: {blur_score:.1f} (higher = sharper)")

    # Get second frame for motion analysis
    batch2 = next(processor.stream_frames(batch_size=1, start_frame=1, end_frame=2))
    frame2 = batch2.frames[0]

    # Detect scene change
    is_scene_change = analyzer.detect_scene_change(frame, frame2, threshold=30.0)
    print(f"\nScene change between frame 0 and 1: {is_scene_change}")

    # Calculate motion
    motion = analyzer.calculate_motion(frame, frame2)
    print(f"Motion magnitude: {motion:.2f} pixels")

    # Example 5: Batch preprocessing
    print("\n" + "=" * 60)
    print("Example 5: Batch Preprocessing")
    print("=" * 60)

    # Reset processor
    processor.seek(0)

    # Process multiple batches
    batch_count = 0
    for batch in processor.stream_frames(batch_size=8):
        processed_batch = preprocessor.process_batch(batch)

        print(f"\nBatch {batch_count}:")
        print(f"  Input frames: {batch.batch_size}")
        print(f"  Output frames: {processed_batch.batch_size}")
        print(f"  Shape: {processed_batch.frames[0].shape}")

        batch_count += 1
        if batch_count >= 3:  # Process first 3 batches
            break

    # Example 6: Custom preprocessing
    print("\n" + "=" * 60)
    print("Example 6: Custom Preprocessing Pipeline")
    print("=" * 60)

    # Custom config with crop and resize
    custom_config = PreprocessConfig(
        crop_box=(100, 100, metadata.width - 100, metadata.height - 100),
        target_size=(512, 512),
        normalize=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    custom_preprocessor = FramePreprocessor(custom_config)

    processor.seek(0)
    batch = next(processor.stream_frames(batch_size=1))
    frame = batch.frames[0]

    print(f"\nOriginal: {frame.shape}")
    print(f"Crop box: {custom_config.crop_box}")
    print(f"Target size: {custom_config.target_size}")

    processed_custom = custom_preprocessor.process_frame(frame)

    print(f"Result: {processed_custom.shape}")
    print(f"Value range: [{processed_custom.min():.3f}, {processed_custom.max():.3f}]")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
