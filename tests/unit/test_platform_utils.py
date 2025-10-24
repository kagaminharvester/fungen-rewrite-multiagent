"""
Unit tests for platform_utils.py

Tests hardware detection, configuration profiles, and performance optimization
across different platforms (Pi, RTX 3090, CPU-only).
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.platform_utils import (
    HardwareInfo,
    HardwareType,
    PerformanceConfig,
    PlatformDetector,
    PlatformProfile,
    detect_hardware,
    get_device,
    get_performance_config,
)


class TestPlatformDetector(unittest.TestCase):
    """Tests for PlatformDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = PlatformDetector(verbose=False)

    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertIsNone(self.detector._cache)

    def test_cpu_info_extraction(self):
        """Test CPU information extraction."""
        cpu_info = self.detector._get_cpu_info()

        self.assertIn("processor", cpu_info)
        self.assertIn("architecture", cpu_info)
        self.assertIn("system", cpu_info)
        self.assertIn("is_arm", cpu_info)
        self.assertIn("cpu_count", cpu_info)

        # CPU count should be positive integer
        self.assertGreater(cpu_info["cpu_count"], 0)

    @patch(
        "builtins.open", mock_open(read_data="Model : Raspberry Pi 4 Model B\nHardware : BCM2835\n")
    )
    def test_raspberry_pi_detection(self):
        """Test Raspberry Pi detection from /proc/cpuinfo."""
        cpu_info = self.detector._get_cpu_info()

        # Should detect Pi
        self.assertTrue(cpu_info.get("is_raspberry_pi", False))

    def test_hardware_type_detection(self):
        """Test hardware type detection."""
        hw_type = self.detector._detect_gpu_type()

        # Should return a valid HardwareType
        self.assertIsInstance(hw_type, HardwareType)
        self.assertIn(
            hw_type, [HardwareType.CUDA, HardwareType.ROCM, HardwareType.CPU, HardwareType.UNKNOWN]
        )

    def test_device_string_generation(self):
        """Test PyTorch device string generation."""
        device = self.detector.get_device_string(0)

        # Should return valid device string
        self.assertIsInstance(device, str)
        self.assertTrue(
            device.startswith("cuda:") or device == "cpu", f"Invalid device string: {device}"
        )

    def test_performance_config_cpu(self):
        """Test performance configuration for CPU-only."""
        # Mock CPU-only environment
        with patch.object(self.detector, "detect_hardware") as mock_detect:
            mock_detect.return_value = HardwareInfo(
                hardware_type=HardwareType.CPU,
                device_name="CPU",
                device_count=0,
                total_memory_gb=0.0,
                available_memory_gb=0.0,
                compute_capability=None,
                driver_version=None,
                cuda_version=None,
                rocm_version=None,
                cpu_info={"cpu_count": 4, "is_raspberry_pi": True},
                platform_profile=PlatformProfile.DEV_PI,
                supports_tensorrt=False,
                supports_fp16=False,
                supports_int8=False,
            )

            config = self.detector.get_performance_config()

            # CPU config should be conservative
            self.assertEqual(config.batch_size, 1)
            self.assertEqual(config.num_workers, 1)
            self.assertFalse(config.use_tensorrt)
            self.assertFalse(config.use_fp16)
            self.assertFalse(config.enable_optical_flow)
            self.assertFalse(config.enable_reid)
            self.assertEqual(config.target_fps, 5)

    def test_performance_config_gpu(self):
        """Test performance configuration for GPU."""
        # Mock RTX 3090 environment
        with patch.object(self.detector, "detect_hardware") as mock_detect:
            mock_detect.return_value = HardwareInfo(
                hardware_type=HardwareType.CUDA,
                device_name="NVIDIA GeForce RTX 3090",
                device_count=1,
                total_memory_gb=24.0,
                available_memory_gb=22.0,
                compute_capability=(8, 6),
                driver_version="535.0",
                cuda_version="12.1",
                rocm_version=None,
                cpu_info={"cpu_count": 16},
                platform_profile=PlatformProfile.PROD_RTX3090,
                supports_tensorrt=True,
                supports_fp16=True,
                supports_int8=True,
            )

            config = self.detector.get_performance_config()

            # GPU config should be aggressive
            self.assertGreater(config.batch_size, 1)
            self.assertGreaterEqual(config.num_workers, 3)
            self.assertTrue(config.use_tensorrt)
            self.assertTrue(config.use_fp16)
            self.assertTrue(config.enable_optical_flow)
            self.assertTrue(config.enable_reid)
            self.assertEqual(config.target_fps, 100)

    def test_batch_size_optimization(self):
        """Test batch size optimization based on VRAM."""
        # Mock GPU with 24GB VRAM
        with patch.object(self.detector, "detect_hardware") as mock_detect:
            mock_detect.return_value = HardwareInfo(
                hardware_type=HardwareType.CUDA,
                device_name="RTX 3090",
                device_count=1,
                total_memory_gb=24.0,
                available_memory_gb=20.0,
                compute_capability=(8, 6),
                driver_version="535.0",
                cuda_version="12.1",
                rocm_version=None,
                cpu_info={},
                platform_profile=PlatformProfile.PROD_RTX3090,
                supports_tensorrt=True,
                supports_fp16=True,
                supports_int8=True,
            )

            # Test with small model
            batch_size = self.detector.optimize_batch_size(model_vram_gb=2.0, frame_vram_gb=0.5)
            self.assertGreater(batch_size, 1)
            self.assertLessEqual(batch_size, 8)

            # Test with large model (should reduce batch size)
            batch_size_large = self.detector.optimize_batch_size(
                model_vram_gb=15.0, frame_vram_gb=0.5
            )
            self.assertLess(batch_size_large, batch_size)

    def test_platform_profile_detection(self):
        """Test automatic platform profile detection."""
        # Test Pi detection
        profile = self.detector._determine_platform_profile(
            HardwareType.CPU, "CPU", {"is_raspberry_pi": True, "cpu_count": 4}
        )
        self.assertEqual(profile, PlatformProfile.DEV_PI)

        # Test RTX 3090 detection
        profile = self.detector._determine_platform_profile(
            HardwareType.CUDA,
            "NVIDIA GeForce RTX 3090",
            {"is_raspberry_pi": False, "cpu_count": 16},
        )
        self.assertEqual(profile, PlatformProfile.PROD_RTX3090)

        # Test generic CUDA GPU
        profile = self.detector._determine_platform_profile(
            HardwareType.CUDA, "Some GPU", {"is_raspberry_pi": False, "cpu_count": 8}
        )
        self.assertEqual(profile, PlatformProfile.PROD_RTX3090)

    def test_config_export(self):
        """Test configuration export to JSON."""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config.json"
            self.detector.export_config(output_path)

            # Verify file exists
            self.assertTrue(output_path.exists())

            # Verify JSON is valid
            with open(output_path) as f:
                data = json.load(f)

            self.assertIn("hardware", data)
            self.assertIn("cpu", data)
            self.assertIn("performance_config", data)

    def test_caching(self):
        """Test hardware detection caching."""
        # First call should detect and cache
        hw1 = self.detector.detect_hardware()
        self.assertIsNotNone(self.detector._cache)

        # Second call should return cached value
        hw2 = self.detector.detect_hardware()
        self.assertIs(hw1, hw2)

        # Force refresh should re-detect
        hw3 = self.detector.detect_hardware(force_refresh=True)
        self.assertIsNotNone(hw3)


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for module-level convenience functions."""

    def test_detect_hardware_function(self):
        """Test detect_hardware convenience function."""
        hw_info = detect_hardware()
        self.assertIsInstance(hw_info, HardwareInfo)

    def test_get_device_function(self):
        """Test get_device convenience function."""
        device = get_device()
        self.assertIsInstance(device, str)
        self.assertTrue(device.startswith("cuda:") or device == "cpu")

        # Test CPU preference
        cpu_device = get_device(prefer_gpu=False)
        self.assertEqual(cpu_device, "cpu")

    def test_get_performance_config_function(self):
        """Test get_performance_config convenience function."""
        config = get_performance_config()
        self.assertIsInstance(config, PerformanceConfig)

        # Test with custom resolution
        config_4k = get_performance_config((3840, 2160))
        self.assertIsInstance(config_4k, PerformanceConfig)


class TestEnvironmentVariables(unittest.TestCase):
    """Tests for environment variable configuration."""

    def test_profile_override(self):
        """Test FUNGEN_PROFILE environment variable."""
        detector = PlatformDetector(verbose=False)

        # Test dev_pi override
        with patch.dict(os.environ, {"FUNGEN_PROFILE": "dev_pi"}):
            profile = detector._determine_platform_profile(
                HardwareType.CUDA, "RTX 3090", {}  # Even with CUDA
            )
            self.assertEqual(profile, PlatformProfile.DEV_PI)

        # Test prod_rtx3090 override
        with patch.dict(os.environ, {"FUNGEN_PROFILE": "prod_rtx3090"}):
            profile = detector._determine_platform_profile(
                HardwareType.CPU, "CPU", {}  # Even with CPU
            )
            self.assertEqual(profile, PlatformProfile.PROD_RTX3090)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_zero_vram_batch_size(self):
        """Test batch size calculation with insufficient VRAM."""
        detector = PlatformDetector(verbose=False)

        with patch.object(detector, "detect_hardware") as mock_detect:
            mock_detect.return_value = HardwareInfo(
                hardware_type=HardwareType.CUDA,
                device_name="Low VRAM GPU",
                device_count=1,
                total_memory_gb=4.0,
                available_memory_gb=2.0,  # Insufficient
                compute_capability=(7, 0),
                driver_version="535.0",
                cuda_version="12.1",
                rocm_version=None,
                cpu_info={},
                platform_profile=PlatformProfile.PROD_RTX3090,
                supports_tensorrt=False,
                supports_fp16=True,
                supports_int8=False,
            )

            # Should return minimum batch size
            with self.assertWarns(UserWarning):
                batch_size = detector.optimize_batch_size(
                    model_vram_gb=5.0, frame_vram_gb=0.5  # More than available
                )
            self.assertEqual(batch_size, 1)

    def test_cpu_only_batch_size(self):
        """Test batch size calculation for CPU-only."""
        detector = PlatformDetector(verbose=False)

        with patch.object(detector, "detect_hardware") as mock_detect:
            mock_detect.return_value = HardwareInfo(
                hardware_type=HardwareType.CPU,
                device_name="CPU",
                device_count=0,
                total_memory_gb=0.0,
                available_memory_gb=0.0,
                compute_capability=None,
                driver_version=None,
                cuda_version=None,
                rocm_version=None,
                cpu_info={},
                platform_profile=PlatformProfile.DEV_PI,
                supports_tensorrt=False,
                supports_fp16=False,
                supports_int8=False,
            )

            # CPU should always return batch size 1
            batch_size = detector.optimize_batch_size(model_vram_gb=0.0, frame_vram_gb=0.5)
            self.assertEqual(batch_size, 1)


if __name__ == "__main__":
    unittest.main()
