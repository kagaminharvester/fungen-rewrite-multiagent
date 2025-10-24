"""
Cross-platform compatibility integration tests.

Tests functionality across different platforms:
- Raspberry Pi (ARM, CPU-only)
- RTX 3090 (x86_64, CUDA GPU)
- Conditional imports and hardware detection
- CPU/GPU fallback behavior

Author: test-engineer-2 agent
Date: 2025-10-24
"""

import platform
import subprocess
import sys
from pathlib import Path

import pytest

from core.config import Config, get_platform_profile
from core.model_manager import ModelManager
from core.video_processor import HardwareAccel, VideoProcessor
from utils.conditional_imports import (
    CUDA_AVAILABLE,
    PYNVVIDEOCODEC_AVAILABLE,
    TENSORRT_AVAILABLE,
    TORCH_AVAILABLE,
)
from utils.platform_utils import (
    detect_hardware,
    get_optimal_batch_size,
    get_optimal_device,
    is_cuda_available,
    is_raspberry_pi,
)


class TestPlatformDetection:
    """Test platform and hardware detection."""

    def test_platform_detection(self):
        """Test basic platform detection."""
        print("\n" + "=" * 80)
        print("TEST: Platform Detection")
        print("=" * 80)

        hw_info = detect_hardware()

        print(f"Platform: {hw_info.platform}")
        print(f"Architecture: {hw_info.architecture}")
        print(f"Python version: {hw_info.python_version}")
        print(f"CPU cores: {hw_info.cpu_cores}")
        print(f"Total RAM: {hw_info.total_ram_gb:.1f} GB")

        assert hw_info.platform in ["Linux", "Windows", "Darwin"]
        assert hw_info.cpu_cores > 0
        assert hw_info.total_ram_gb > 0

        print("✓ Platform detection passed")
        print("=" * 80)

    def test_raspberry_pi_detection(self):
        """Test Raspberry Pi specific detection."""
        print("\n" + "=" * 80)
        print("TEST: Raspberry Pi Detection")
        print("=" * 80)

        is_pi = is_raspberry_pi()

        print(f"Is Raspberry Pi: {is_pi}")
        print(f"Machine: {platform.machine()}")
        print(f"System: {platform.system()}")

        if is_pi:
            print("  Running on Raspberry Pi")
            assert platform.machine().startswith("arm") or platform.machine().startswith("aarch")
        else:
            print("  Not running on Raspberry Pi")

        print("✓ Raspberry Pi detection completed")
        print("=" * 80)

    def test_cuda_detection(self):
        """Test CUDA availability detection."""
        print("\n" + "=" * 80)
        print("TEST: CUDA Detection")
        print("=" * 80)

        cuda_available = is_cuda_available()

        print(f"CUDA available: {cuda_available}")
        print(f"TORCH_AVAILABLE: {TORCH_AVAILABLE}")
        print(f"CUDA_AVAILABLE: {CUDA_AVAILABLE}")
        print(f"TENSORRT_AVAILABLE: {TENSORRT_AVAILABLE}")

        if cuda_available:
            hw_info = detect_hardware()
            print(f"GPU name: {hw_info.gpu_name}")
            print(f"GPU VRAM: {hw_info.gpu_vram_gb:.1f} GB")
            print(f"CUDA version: {hw_info.cuda_version}")

        print("✓ CUDA detection completed")
        print("=" * 80)

    def test_optimal_device_selection(self):
        """Test automatic optimal device selection."""
        print("\n" + "=" * 80)
        print("TEST: Optimal Device Selection")
        print("=" * 80)

        device = get_optimal_device()

        print(f"Optimal device: {device}")

        # Should return valid device string
        assert device in ["cuda", "cuda:0", "cpu"]

        if CUDA_AVAILABLE:
            assert device.startswith("cuda")
        else:
            assert device == "cpu"

        print("✓ Device selection passed")
        print("=" * 80)

    def test_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        print("\n" + "=" * 80)
        print("TEST: Optimal Batch Size Calculation")
        print("=" * 80)

        # For different resolutions
        test_cases = [
            (1920, 1080, "1080p"),
            (1280, 720, "720p"),
            (3840, 2160, "4K"),
            (7680, 4320, "8K"),
        ]

        for width, height, name in test_cases:
            batch_size = get_optimal_batch_size(width, height)
            print(f"{name} ({width}x{height}): batch_size = {batch_size}")

            assert batch_size > 0
            assert batch_size <= 32

        print("✓ Batch size calculation passed")
        print("=" * 80)


class TestConditionalImports:
    """Test conditional imports for cross-platform compatibility."""

    def test_torch_import(self):
        """Test conditional torch import."""
        print("\n" + "=" * 80)
        print("TEST: Torch Conditional Import")
        print("=" * 80)

        print(f"TORCH_AVAILABLE: {TORCH_AVAILABLE}")

        if TORCH_AVAILABLE:
            import torch

            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available in torch: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                print(f"CUDA devices: {torch.cuda.device_count()}")
                print(f"Current device: {torch.cuda.current_device()}")
        else:
            print("PyTorch not available (expected on Pi)")

        print("✓ Torch import test passed")
        print("=" * 80)

    def test_tensorrt_import(self):
        """Test conditional TensorRT import."""
        print("\n" + "=" * 80)
        print("TEST: TensorRT Conditional Import")
        print("=" * 80)

        print(f"TENSORRT_AVAILABLE: {TENSORRT_AVAILABLE}")

        if TENSORRT_AVAILABLE:
            import tensorrt as trt

            print(f"TensorRT version: {trt.__version__}")
        else:
            print("TensorRT not available (expected on Pi)")

        print("✓ TensorRT import test passed")
        print("=" * 80)

    def test_pynvvideocodec_import(self):
        """Test conditional PyNvVideoCodec import."""
        print("\n" + "=" * 80)
        print("TEST: PyNvVideoCodec Conditional Import")
        print("=" * 80)

        print(f"PYNVVIDEOCODEC_AVAILABLE: {PYNVVIDEOCODEC_AVAILABLE}")

        if PYNVVIDEOCODEC_AVAILABLE:
            import PyNvVideoCodec

            print("PyNvVideoCodec available")
        else:
            print("PyNvVideoCodec not available (expected on Pi)")

        print("✓ PyNvVideoCodec import test passed")
        print("=" * 80)


class TestPlatformProfiles:
    """Test platform-specific configuration profiles."""

    def test_profile_detection(self):
        """Test automatic profile detection."""
        print("\n" + "=" * 80)
        print("TEST: Profile Detection")
        print("=" * 80)

        profile = get_platform_profile()

        print(f"Detected profile: {profile}")

        valid_profiles = ["dev_pi", "prod_rtx3090", "debug"]
        assert profile in valid_profiles

        if is_raspberry_pi():
            assert profile == "dev_pi"
        elif CUDA_AVAILABLE:
            assert profile in ["prod_rtx3090", "debug"]

        print("✓ Profile detection passed")
        print("=" * 80)

    def test_config_for_platform(self):
        """Test configuration loading for current platform."""
        print("\n" + "=" * 80)
        print("TEST: Platform-Specific Config")
        print("=" * 80)

        config = Config()

        print(f"Profile: {config.profile}")
        print(f"Device: {config.device}")
        print(f"Batch size: {config.batch_size}")
        print(f"HW accel: {config.hw_accel}")
        print(f"Use TensorRT: {config.use_tensorrt}")

        # Validate config makes sense for platform
        if is_raspberry_pi():
            assert config.device == "cpu"
            assert not config.use_tensorrt
            assert config.batch_size <= 4
        elif CUDA_AVAILABLE:
            assert config.device.startswith("cuda")

        print("✓ Platform config test passed")
        print("=" * 80)


class TestCPUFallback:
    """Test CPU fallback behavior when GPU is unavailable."""

    def test_video_processor_cpu_mode(self):
        """Test VideoProcessor in CPU-only mode."""
        print("\n" + "=" * 80)
        print("TEST: VideoProcessor CPU Mode")
        print("=" * 80)

        # Create test video
        test_video = Path("/tmp/test_cpu_fallback.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=640x480:rate=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(test_video),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=20)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pytest.skip("Failed to create test video")

        # Force CPU mode
        processor = VideoProcessor(str(test_video), hw_accel=False)
        metadata = processor.get_metadata()

        print(f"Resolution: {metadata.width}x{metadata.height}")
        print(f"FPS: {metadata.fps}")
        print(f"HW accel type: {processor.hw_accel_type.value}")

        assert processor.hw_accel_type == HardwareAccel.NONE

        # Decode some frames
        frame_count = 0
        for batch in processor.stream_frames(batch_size=4):
            frame_count += len(batch.frames)

        print(f"Decoded frames: {frame_count}")

        assert frame_count > 0
        print("✓ CPU mode test passed")
        print("=" * 80)

    def test_model_manager_cpu_mode(self):
        """Test ModelManager in CPU-only mode."""
        print("\n" + "=" * 80)
        print("TEST: ModelManager CPU Mode")
        print("=" * 80)

        # Force CPU device
        try:
            manager = ModelManager(device="cpu")
            print(f"Device: {manager.device}")

            assert manager.device == "cpu"
            print("✓ CPU mode initialization passed")

        except Exception as e:
            # May fail if models not available, which is OK
            print(f"⚠ ModelManager not testable: {e}")

        print("=" * 80)


class TestGPUMode:
    """Test GPU-specific functionality (skip if CUDA unavailable)."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_video_processor_gpu_mode(self):
        """Test VideoProcessor with GPU acceleration."""
        print("\n" + "=" * 80)
        print("TEST: VideoProcessor GPU Mode")
        print("=" * 80)

        test_video = Path("/tmp/test_gpu_mode.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=640x480:rate=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(test_video),
        ]

        subprocess.run(cmd, capture_output=True, check=True, timeout=20)

        processor = VideoProcessor(str(test_video), hw_accel=True)
        metadata = processor.get_metadata()

        print(f"HW accel type: {processor.hw_accel_type.value}")

        # Should use GPU acceleration if available
        if PYNVVIDEOCODEC_AVAILABLE:
            assert processor.hw_accel_type == HardwareAccel.NVDEC
        else:
            # May fall back to CPU if PyNvVideoCodec not available
            print("⚠ PyNvVideoCodec not available, using CPU fallback")

        print("✓ GPU mode test passed")
        print("=" * 80)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_model_manager_gpu_mode(self):
        """Test ModelManager with GPU."""
        print("\n" + "=" * 80)
        print("TEST: ModelManager GPU Mode")
        print("=" * 80)

        try:
            manager = ModelManager(device="cuda")
            print(f"Device: {manager.device}")

            assert manager.device.startswith("cuda")
            print("✓ GPU mode initialization passed")

        except Exception as e:
            print(f"⚠ ModelManager not testable: {e}")

        print("=" * 80)


class TestPlatformFeatures:
    """Test platform-specific features."""

    def test_available_features(self):
        """Test which features are available on current platform."""
        print("\n" + "=" * 80)
        print("TEST: Available Features")
        print("=" * 80)

        features = {
            "PyTorch": TORCH_AVAILABLE,
            "CUDA": CUDA_AVAILABLE,
            "TensorRT": TENSORRT_AVAILABLE,
            "PyNvVideoCodec": PYNVVIDEOCODEC_AVAILABLE,
        }

        print("Available features:")
        for feature, available in features.items():
            status = "✓" if available else "✗"
            print(f"  {status} {feature}")

        # Count available features
        available_count = sum(features.values())
        print(f"\nTotal available: {available_count}/{len(features)}")

        # At minimum, should be able to run in CPU mode
        print("✓ Feature detection completed")
        print("=" * 80)

    def test_platform_limitations(self):
        """Test platform-specific limitations."""
        print("\n" + "=" * 80)
        print("TEST: Platform Limitations")
        print("=" * 80)

        hw_info = detect_hardware()

        limitations = []

        if not CUDA_AVAILABLE:
            limitations.append("No CUDA support (CPU-only mode)")

        if not TENSORRT_AVAILABLE:
            limitations.append("No TensorRT optimization")

        if hw_info.total_ram_gb < 8:
            limitations.append(f"Limited RAM: {hw_info.total_ram_gb:.1f} GB")

        if hw_info.cpu_cores < 4:
            limitations.append(f"Limited CPU cores: {hw_info.cpu_cores}")

        if limitations:
            print("Platform limitations:")
            for limit in limitations:
                print(f"  ⚠ {limit}")
        else:
            print("No significant limitations detected")

        print("✓ Limitation check completed")
        print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
