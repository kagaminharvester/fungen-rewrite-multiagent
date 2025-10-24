#!/usr/bin/env python3
"""
FunGen Rewrite - Main Entry Point

This is the primary entry point for the FunGen video tracking and funscript generation system.
Supports both CLI mode (batch processing) and GUI mode (interactive).

Usage:
    # GUI mode (default)
    python main.py

    # CLI mode - single video
    python main.py --cli video.mp4 --output output.funscript

    # CLI mode - batch processing
    python main.py --cli --batch videos/ --output output/

    # Specify tracker and model
    python main.py --cli video.mp4 --tracker improved --model yolo11n

    # Hardware profile selection
    python main.py --profile prod_rtx3090  # or dev_pi, debug

Author: integration-master agent
Date: 2025-10-24
Version: 1.0.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("fungen.log")],
)

logger = logging.getLogger(__name__)


def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="FunGen - AI-Powered Funscript Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI
  python main.py

  # Process single video
  python main.py --cli video.mp4 -o output.funscript

  # Batch process directory
  python main.py --cli --batch videos/ -o output/

  # Specify tracker and model
  python main.py --cli video.mp4 --tracker improved --model yolo11n

  # Select hardware profile
  python main.py --profile prod_rtx3090

For more information, see docs/README.md
        """,
    )

    # Mode selection
    parser.add_argument(
        "--cli",
        "--batch",
        action="store_true",
        dest="cli_mode",
        help="Run in CLI mode (default: GUI)",
    )

    parser.add_argument("--gui", action="store_true", help="Explicitly launch GUI mode (default)")

    # Input/output
    parser.add_argument(
        "input", nargs="?", type=str, help="Input video file or directory (for CLI mode)"
    )

    parser.add_argument("-o", "--output", type=str, help="Output file or directory path")

    parser.add_argument(
        "--batch", type=str, dest="batch_dir", help="Process all videos in directory (CLI mode)"
    )

    # Tracker selection
    parser.add_argument(
        "--tracker",
        type=str,
        default="improved",
        choices=["bytetrack", "improved", "hybrid"],
        help="Tracker algorithm (default: improved)",
    )

    # Model selection
    parser.add_argument(
        "--model", type=str, default="yolo11n", help="YOLO model name (default: yolo11n)"
    )

    # Hardware/performance
    parser.add_argument(
        "--profile",
        type=str,
        choices=["dev_pi", "prod_rtx3090", "debug", "auto"],
        default="auto",
        help="Hardware profile (default: auto-detect)",
    )

    parser.add_argument("--device", type=str, help="Device string (cuda:0, cpu, etc.)")

    parser.add_argument("--batch-size", type=int, help="Inference batch size (default: auto)")

    parser.add_argument("--workers", type=int, help="Number of worker processes (default: auto)")

    # Tracker parameters
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for matching (default: 0.45)",
    )

    # Features
    parser.add_argument("--no-tensorrt", action="store_true", help="Disable TensorRT optimization")

    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 precision")

    parser.add_argument(
        "--no-optical-flow", action="store_true", help="Disable optical flow refinement"
    )

    # Logging/debugging
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument("--version", action="version", version="FunGen Rewrite 1.0.0")

    return parser


def run_cli_mode(args: argparse.Namespace) -> int:
    """Run in CLI mode for batch processing.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        from core.batch_processor import BatchProcessor, ProcessingSettings
        from core.config import Config
        from core.model_manager import ModelManager
        from core.video_processor import VideoProcessor
        from trackers import ByteTracker
        from trackers.improved_tracker import ImprovedTracker

        logger.info("FunGen CLI Mode")
        logger.info("=" * 60)

        # Load configuration
        if args.profile != "auto":
            config = Config.from_profile(args.profile)
        else:
            config = Config.auto_detect()

        logger.info(f"Configuration: {config}")

        # Override settings from CLI args
        if args.device:
            config.device = args.device
        if args.batch_size:
            config.max_batch_size = args.batch_size
        if args.workers:
            config.num_workers = args.workers
        if args.no_tensorrt:
            config.use_tensorrt = False
        if args.no_fp16:
            config.use_fp16 = False
        if args.no_optical_flow:
            config.enable_optical_flow = False

        # Get input files
        input_files = []
        if args.batch_dir:
            input_dir = Path(args.batch_dir)
            if not input_dir.exists():
                logger.error(f"Directory not found: {input_dir}")
                return 1
            # Find video files
            for ext in ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm"]:
                input_files.extend(input_dir.glob(ext))
        elif args.input:
            input_path = Path(args.input)
            if not input_path.exists():
                logger.error(f"File not found: {input_path}")
                return 1
            input_files = [input_path]
        else:
            logger.error("No input specified. Use --help for usage.")
            return 1

        if not input_files:
            logger.error("No video files found.")
            return 1

        logger.info(f"Found {len(input_files)} video(s) to process")

        # Setup output directory
        output_dir = Path(args.output) if args.output else Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model manager
        logger.info(f"Loading model: {args.model}")
        model_manager = ModelManager(
            model_dir=config.model_dir,
            device=config.device,
            max_batch_size=config.max_batch_size,
            verbose=args.verbose,
        )

        if not model_manager.load_model(args.model, optimize=config.use_tensorrt):
            logger.error("Failed to load model")
            return 1

        logger.info(f"Model loaded: {model_manager}")

        # Initialize tracker
        tracker_class = ImprovedTracker if args.tracker == "improved" else ByteTracker
        logger.info(f"Using tracker: {args.tracker}")

        # Process each video
        for video_path in input_files:
            try:
                logger.info(f"\nProcessing: {video_path.name}")
                logger.info("-" * 60)

                # Initialize video processor
                processor = VideoProcessor(str(video_path), hw_accel=True)
                metadata = processor.get_metadata()
                logger.info(f"Video: {metadata.width}x{metadata.height} @ {metadata.fps:.2f} FPS")
                logger.info(
                    f"Duration: {metadata.duration_sec:.2f}s ({metadata.total_frames} frames)"
                )

                # Initialize tracker
                tracker = tracker_class(
                    max_age=30,
                    min_hits=3,
                    iou_threshold=args.iou_threshold,
                    use_optical_flow=config.enable_optical_flow and not args.no_optical_flow,
                )

                # Process video
                frame_count = 0
                import time

                start_time = time.time()

                for batch in processor.stream_frames(batch_size=config.max_batch_size):
                    # Run inference
                    detections_batch = model_manager.predict_batch(
                        batch.frames,
                        conf_threshold=args.conf_threshold,
                        iou_threshold=args.iou_threshold,
                    )

                    # Update tracker with each frame's detections
                    for frame_detections in detections_batch:
                        # Convert from ModelManager Detection to Tracker Detection
                        from trackers.base_tracker import Detection

                        tracker_detections = [
                            Detection(
                                bbox=d.bbox,
                                confidence=d.confidence,
                                class_id=d.class_id,
                                class_name=d.class_name,
                                frame_id=frame_count,
                            )
                            for d in frame_detections
                        ]

                        if frame_count == 0:
                            tracker.initialize(tracker_detections)
                        else:
                            tracker.update(tracker_detections)

                        frame_count += 1

                    # Progress logging
                    if frame_count % 100 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        progress = (frame_count / metadata.total_frames) * 100
                        logger.info(
                            f"Progress: {progress:.1f}% ({frame_count}/{metadata.total_frames}) - {fps:.1f} FPS"
                        )

                # Generate funscript
                funscript_data = tracker.get_funscript_data(
                    axis="vertical", frame_height=metadata.height, fps=metadata.fps
                )

                # Save funscript
                output_path = output_dir / f"{video_path.stem}.funscript"
                with open(output_path, "w") as f:
                    json.dump(funscript_data.to_dict(), f, indent=2)

                # Final stats
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"\nCompleted in {elapsed:.2f}s (avg {avg_fps:.1f} FPS)")
                logger.info(f"Output: {output_path}")
                logger.info(f"Actions generated: {len(funscript_data.actions)}")

            except Exception as e:
                logger.error(f"Error processing {video_path.name}: {e}", exc_info=True)
                continue

        logger.info("\n" + "=" * 60)
        logger.info("Batch processing complete!")
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


def run_gui_mode(args: argparse.Namespace) -> int:
    """Run in GUI mode.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        import tkinter as tk

        from core.config import Config
        from ui.main_window import MainWindow

        logger.info("FunGen GUI Mode")

        # Load configuration
        if args.profile != "auto":
            config = Config.from_profile(args.profile)
        else:
            config = Config.auto_detect()

        # Create and run GUI
        root = tk.Tk()
        app = MainWindow(root, config)
        root.mainloop()

        return 0

    except ImportError as e:
        logger.error(f"GUI dependencies not available: {e}")
        logger.error("Install with: pip install tkinter sv-ttk")
        return 1
    except Exception as e:
        logger.error(f"GUI error: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # Print banner
    print(
        """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   FunGen Rewrite - AI-Powered Funscript Generator        ║
    ║   Version 1.0.0                                           ║
    ║                                                           ║
    ║   Target: 100+ FPS (RTX 3090) | Cross-platform support   ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    )

    # Run in appropriate mode
    if args.cli_mode or args.input or args.batch_dir:
        return run_cli_mode(args)
    else:
        return run_gui_mode(args)


if __name__ == "__main__":
    sys.exit(main())
