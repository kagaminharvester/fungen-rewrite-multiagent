"""
Cross-platform hardware detection and configuration utilities.

This module provides comprehensive hardware detection for CUDA, ROCm, and CPU-only
environments, enabling seamless operation on both Raspberry Pi (development) and
RTX 3090 (production) platforms.

Target Performance:
- Raspberry Pi: 5+ FPS CPU mode
- RTX 3090: 100+ FPS GPU mode with TensorRT FP16
"""

import json
import os
import platform
import subprocess
import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class HardwareType(Enum):
    """Hardware acceleration types."""

    CUDA = "cuda"
    ROCM = "rocm"
    CPU = "cpu"
    UNKNOWN = "unknown"


class PlatformProfile(Enum):
    """Platform configuration profiles."""

    DEV_PI = "dev_pi"  # Raspberry Pi - CPU only, minimal features
    PROD_RTX3090 = "prod_rtx3090"  # RTX 3090 - Full GPU, TensorRT FP16
    DEBUG = "debug"  # Development with extensive logging
    AUTO = "auto"  # Automatic detection


@dataclass
class HardwareInfo:
    """Comprehensive hardware information."""

    hardware_type: HardwareType
    device_name: str
    device_count: int
    total_memory_gb: float
    available_memory_gb: float
    compute_capability: Optional[Tuple[int, int]]
    driver_version: Optional[str]
    cuda_version: Optional[str]
    rocm_version: Optional[str]
    cpu_info: Dict[str, Any]
    platform_profile: PlatformProfile
    supports_tensorrt: bool
    supports_fp16: bool
    supports_int8: bool


@dataclass
class PerformanceConfig:
    """Performance configuration based on hardware."""

    batch_size: int
    num_workers: int
    use_tensorrt: bool
    use_fp16: bool
    enable_optical_flow: bool
    enable_reid: bool
    max_video_resolution: Tuple[int, int]
    target_fps: int
    vram_limit_gb: float


class PlatformDetector:
    """Hardware detection and platform configuration manager."""

    def __init__(self, verbose: bool = True):
        """
        Initialize platform detector.

        Args:
            verbose: Enable detailed logging during detection
        """
        self.verbose = verbose
        self._cache: Optional[HardwareInfo] = None

    def detect_hardware(self, force_refresh: bool = False) -> HardwareInfo:
        """
        Detect available hardware and capabilities.

        Args:
            force_refresh: Bypass cache and re-detect hardware

        Returns:
            HardwareInfo object with complete hardware details
        """
        if self._cache is not None and not force_refresh:
            return self._cache

        if self.verbose:
            print("Detecting hardware configuration...")

        # Detect GPU type
        hardware_type = self._detect_gpu_type()

        # Get CPU information
        cpu_info = self._get_cpu_info()

        # Initialize default values
        device_name = "CPU"
        device_count = 0
        total_memory_gb = 0.0
        available_memory_gb = 0.0
        compute_capability = None
        driver_version = None
        cuda_version = None
        rocm_version = None
        supports_tensorrt = False
        supports_fp16 = False
        supports_int8 = False

        # Get GPU-specific information
        if hardware_type == HardwareType.CUDA:
            cuda_info = self._get_cuda_info()
            device_name = cuda_info.get("device_name", "Unknown CUDA Device")
            device_count = cuda_info.get("device_count", 0)
            total_memory_gb = cuda_info.get("total_memory_gb", 0.0)
            available_memory_gb = cuda_info.get("available_memory_gb", 0.0)
            compute_capability = cuda_info.get("compute_capability")
            driver_version = cuda_info.get("driver_version")
            cuda_version = cuda_info.get("cuda_version")
            supports_tensorrt = cuda_info.get("supports_tensorrt", False)
            supports_fp16 = cuda_info.get("supports_fp16", False)
            supports_int8 = cuda_info.get("supports_int8", False)

        elif hardware_type == HardwareType.ROCM:
            rocm_info = self._get_rocm_info()
            device_name = rocm_info.get("device_name", "Unknown ROCm Device")
            device_count = rocm_info.get("device_count", 0)
            total_memory_gb = rocm_info.get("total_memory_gb", 0.0)
            available_memory_gb = rocm_info.get("available_memory_gb", 0.0)
            rocm_version = rocm_info.get("rocm_version")
            supports_fp16 = rocm_info.get("supports_fp16", False)

        # Determine platform profile
        platform_profile = self._determine_platform_profile(hardware_type, device_name, cpu_info)

        # Create hardware info object
        hardware_info = HardwareInfo(
            hardware_type=hardware_type,
            device_name=device_name,
            device_count=device_count,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            compute_capability=compute_capability,
            driver_version=driver_version,
            cuda_version=cuda_version,
            rocm_version=rocm_version,
            cpu_info=cpu_info,
            platform_profile=platform_profile,
            supports_tensorrt=supports_tensorrt,
            supports_fp16=supports_fp16,
            supports_int8=supports_int8,
        )

        # Cache the result
        self._cache = hardware_info

        if self.verbose:
            self._print_hardware_summary(hardware_info)

        return hardware_info

    def _detect_gpu_type(self) -> HardwareType:
        """Detect GPU type (CUDA, ROCm, or CPU-only)."""
        # Try CUDA first
        try:
            import torch

            if torch.cuda.is_available():
                return HardwareType.CUDA
        except ImportError:
            pass
        except Exception as e:
            if self.verbose:
                print(f"CUDA detection warning: {e}")

        # Try ROCm
        try:
            import torch

            if hasattr(torch, "hip") and torch.hip.is_available():
                return HardwareType.ROCM
        except ImportError:
            pass
        except Exception as e:
            if self.verbose:
                print(f"ROCm detection warning: {e}")

        # Check for nvidia-smi
        try:
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and "GPU" in result.stdout:
                # NVIDIA GPU present but PyTorch CUDA not available
                if self.verbose:
                    print("Warning: NVIDIA GPU detected but PyTorch CUDA not available")
                return HardwareType.CUDA
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check for rocm-smi
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return HardwareType.ROCM
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Default to CPU
        return HardwareType.CPU

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        cpu_info = {
            "processor": platform.processor(),
            "architecture": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "python_version": sys.version,
            "is_arm": "arm" in platform.machine().lower() or "aarch" in platform.machine().lower(),
            "is_raspberry_pi": False,
        }

        # Detect Raspberry Pi
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                if "Raspberry Pi" in cpuinfo or "BCM" in cpuinfo:
                    cpu_info["is_raspberry_pi"] = True
                    # Extract Pi model
                    for line in cpuinfo.split("\n"):
                        if "Model" in line:
                            cpu_info["pi_model"] = line.split(":")[-1].strip()
                            break
        except FileNotFoundError:
            pass

        # Get CPU count
        try:
            import multiprocessing

            cpu_info["cpu_count"] = multiprocessing.cpu_count()
        except Exception:
            cpu_info["cpu_count"] = 1

        return cpu_info

    def _get_cuda_info(self) -> Dict[str, Any]:
        """Get CUDA device information."""
        cuda_info = {
            "device_count": 0,
            "device_name": "Unknown",
            "total_memory_gb": 0.0,
            "available_memory_gb": 0.0,
            "compute_capability": None,
            "driver_version": None,
            "cuda_version": None,
            "supports_tensorrt": False,
            "supports_fp16": False,
            "supports_int8": False,
        }

        try:
            import torch

            if torch.cuda.is_available():
                cuda_info["device_count"] = torch.cuda.device_count()
                cuda_info["device_name"] = torch.cuda.get_device_name(0)

                # Memory info (in GB)
                props = torch.cuda.get_device_properties(0)
                cuda_info["total_memory_gb"] = props.total_memory / (1024**3)

                # Available memory
                torch.cuda.empty_cache()
                cuda_info["available_memory_gb"] = torch.cuda.mem_get_info()[0] / (1024**3)

                # Compute capability
                cuda_info["compute_capability"] = (props.major, props.minor)

                # CUDA version
                cuda_info["cuda_version"] = torch.version.cuda

                # Check FP16 support (requires compute capability >= 7.0 for good performance)
                if props.major >= 7:
                    cuda_info["supports_fp16"] = True

                # Check INT8 support (requires compute capability >= 7.5)
                if props.major > 7 or (props.major == 7 and props.minor >= 5):
                    cuda_info["supports_int8"] = True

                # Check TensorRT availability
                try:
                    import tensorrt as trt

                    cuda_info["supports_tensorrt"] = True
                    if self.verbose:
                        print(f"TensorRT version: {trt.__version__}")
                except ImportError:
                    if self.verbose:
                        print("TensorRT not available")

        except Exception as e:
            if self.verbose:
                print(f"Error getting CUDA info: {e}")

        return cuda_info

    def _get_rocm_info(self) -> Dict[str, Any]:
        """Get ROCm device information."""
        rocm_info = {
            "device_count": 0,
            "device_name": "Unknown",
            "total_memory_gb": 0.0,
            "available_memory_gb": 0.0,
            "rocm_version": None,
            "supports_fp16": False,
        }

        try:
            import torch

            if hasattr(torch, "hip") and torch.hip.is_available():
                rocm_info["device_count"] = torch.cuda.device_count()  # Uses same API
                rocm_info["device_name"] = torch.cuda.get_device_name(0)

                # Memory info
                props = torch.cuda.get_device_properties(0)
                rocm_info["total_memory_gb"] = props.total_memory / (1024**3)

                torch.cuda.empty_cache()
                rocm_info["available_memory_gb"] = torch.cuda.mem_get_info()[0] / (1024**3)

                # ROCm version
                if hasattr(torch.version, "hip"):
                    rocm_info["rocm_version"] = torch.version.hip

                # Most modern AMD GPUs support FP16
                rocm_info["supports_fp16"] = True

        except Exception as e:
            if self.verbose:
                print(f"Error getting ROCm info: {e}")

        return rocm_info

    def _determine_platform_profile(
        self, hardware_type: HardwareType, device_name: str, cpu_info: Dict[str, Any]
    ) -> PlatformProfile:
        """Determine the appropriate platform profile."""
        # Check environment variable first
        env_profile = os.getenv("FUNGEN_PROFILE", "").lower()
        if env_profile == "dev_pi":
            return PlatformProfile.DEV_PI
        elif env_profile == "prod_rtx3090":
            return PlatformProfile.PROD_RTX3090
        elif env_profile == "debug":
            return PlatformProfile.DEBUG

        # Auto-detect based on hardware
        if cpu_info.get("is_raspberry_pi", False):
            return PlatformProfile.DEV_PI

        if hardware_type == HardwareType.CUDA:
            if "3090" in device_name or "3080" in device_name or "4090" in device_name:
                return PlatformProfile.PROD_RTX3090
            # Other CUDA GPUs default to production profile
            return PlatformProfile.PROD_RTX3090

        if hardware_type == HardwareType.ROCM:
            return PlatformProfile.PROD_RTX3090  # Treat ROCm similarly to CUDA

        # CPU-only defaults to dev profile
        return PlatformProfile.DEV_PI

    def _print_hardware_summary(self, info: HardwareInfo) -> None:
        """Print hardware detection summary."""
        print("\n" + "=" * 60)
        print("HARDWARE DETECTION SUMMARY")
        print("=" * 60)
        print(f"Platform Profile: {info.platform_profile.value}")
        print(f"Hardware Type: {info.hardware_type.value.upper()}")
        print(f"Device: {info.device_name}")

        if info.device_count > 0:
            print(f"Device Count: {info.device_count}")
            print(f"Total VRAM: {info.total_memory_gb:.2f} GB")
            print(f"Available VRAM: {info.available_memory_gb:.2f} GB")

            if info.compute_capability:
                print(
                    f"Compute Capability: {info.compute_capability[0]}.{info.compute_capability[1]}"
                )

            if info.driver_version:
                print(f"Driver Version: {info.driver_version}")

            if info.cuda_version:
                print(f"CUDA Version: {info.cuda_version}")

            if info.rocm_version:
                print(f"ROCm Version: {info.rocm_version}")

            print(f"TensorRT Support: {'Yes' if info.supports_tensorrt else 'No'}")
            print(f"FP16 Support: {'Yes' if info.supports_fp16 else 'No'}")
            print(f"INT8 Support: {'Yes' if info.supports_int8 else 'No'}")

        print(f"\nCPU: {info.cpu_info.get('processor', 'Unknown')}")
        print(f"Architecture: {info.cpu_info.get('architecture', 'Unknown')}")
        print(f"CPU Cores: {info.cpu_info.get('cpu_count', 'Unknown')}")

        if info.cpu_info.get("is_raspberry_pi"):
            print(f"Raspberry Pi Model: {info.cpu_info.get('pi_model', 'Unknown')}")

        print("=" * 60 + "\n")

    def get_device_string(self, gpu_id: int = 0) -> str:
        """
        Get PyTorch device string.

        Args:
            gpu_id: GPU device ID (0-indexed)

        Returns:
            Device string ('cuda:0', 'cpu', etc.)
        """
        info = self.detect_hardware()

        if info.hardware_type == HardwareType.CUDA:
            return f"cuda:{gpu_id}"
        elif info.hardware_type == HardwareType.ROCM:
            return f"cuda:{gpu_id}"  # ROCm uses same API
        else:
            return "cpu"

    def get_performance_config(
        self, target_resolution: Tuple[int, int] = (1920, 1080)
    ) -> PerformanceConfig:
        """
        Get optimal performance configuration for detected hardware.

        Args:
            target_resolution: Target video resolution (width, height)

        Returns:
            PerformanceConfig optimized for current hardware
        """
        info = self.detect_hardware()

        # Default CPU config (Raspberry Pi)
        if info.platform_profile == PlatformProfile.DEV_PI:
            return PerformanceConfig(
                batch_size=1,
                num_workers=1,
                use_tensorrt=False,
                use_fp16=False,
                enable_optical_flow=False,  # Too slow on CPU
                enable_reid=False,  # Too slow on CPU
                max_video_resolution=(1920, 1080),
                target_fps=5,  # 5+ FPS target on Pi
                vram_limit_gb=0.0,
            )

        # Production GPU config (RTX 3090)
        if info.platform_profile == PlatformProfile.PROD_RTX3090:
            # Calculate optimal batch size based on VRAM
            vram_per_frame_gb = 0.5  # Estimated VRAM per 1080p frame
            resolution_multiplier = (target_resolution[0] * target_resolution[1]) / (1920 * 1080)
            vram_per_frame_gb *= resolution_multiplier

            # Reserve 4GB for model, 2GB for system
            available_for_batch = info.available_memory_gb - 6.0
            optimal_batch = max(1, int(available_for_batch / vram_per_frame_gb))
            optimal_batch = min(optimal_batch, 8)  # Cap at 8 for stability

            # Calculate optimal worker count (3-6 for RTX 3090)
            num_workers = min(6, max(3, info.cpu_info.get("cpu_count", 4) // 4))

            return PerformanceConfig(
                batch_size=optimal_batch,
                num_workers=num_workers,
                use_tensorrt=info.supports_tensorrt,
                use_fp16=info.supports_fp16,
                enable_optical_flow=True,  # GPU-accelerated
                enable_reid=True,  # Full features enabled
                max_video_resolution=(7680, 4320),  # 8K support
                target_fps=100,  # 100+ FPS target
                vram_limit_gb=20.0,  # <20GB VRAM target
            )

        # Debug config
        return PerformanceConfig(
            batch_size=2,
            num_workers=2,
            use_tensorrt=info.supports_tensorrt,
            use_fp16=info.supports_fp16,
            enable_optical_flow=info.hardware_type != HardwareType.CPU,
            enable_reid=info.hardware_type != HardwareType.CPU,
            max_video_resolution=(3840, 2160),  # 4K for debug
            target_fps=30,
            vram_limit_gb=info.total_memory_gb * 0.8,
        )

    def optimize_batch_size(self, model_vram_gb: float, frame_vram_gb: float = 0.5) -> int:
        """
        Calculate optimal batch size for given VRAM requirements.

        Args:
            model_vram_gb: VRAM required for model
            frame_vram_gb: VRAM required per frame

        Returns:
            Optimal batch size
        """
        info = self.detect_hardware()

        if info.hardware_type == HardwareType.CPU:
            return 1

        # Reserve 2GB for system
        available = info.available_memory_gb - 2.0 - model_vram_gb

        if available <= 0:
            warnings.warn(
                f"Insufficient VRAM. Available: {info.available_memory_gb:.2f}GB, "
                f"Required: {model_vram_gb + 2.0:.2f}GB"
            )
            return 1

        batch_size = max(1, int(available / frame_vram_gb))
        return min(batch_size, 8)  # Cap at 8 for stability

    def export_config(self, output_path: Path) -> None:
        """
        Export hardware configuration to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        info = self.detect_hardware()
        config = self.get_performance_config()

        data = {
            "hardware": {
                "type": info.hardware_type.value,
                "device_name": info.device_name,
                "device_count": info.device_count,
                "total_memory_gb": info.total_memory_gb,
                "available_memory_gb": info.available_memory_gb,
                "compute_capability": info.compute_capability,
                "driver_version": info.driver_version,
                "cuda_version": info.cuda_version,
                "rocm_version": info.rocm_version,
                "platform_profile": info.platform_profile.value,
                "supports_tensorrt": info.supports_tensorrt,
                "supports_fp16": info.supports_fp16,
                "supports_int8": info.supports_int8,
            },
            "cpu": info.cpu_info,
            "performance_config": {
                "batch_size": config.batch_size,
                "num_workers": config.num_workers,
                "use_tensorrt": config.use_tensorrt,
                "use_fp16": config.use_fp16,
                "enable_optical_flow": config.enable_optical_flow,
                "enable_reid": config.enable_reid,
                "max_video_resolution": config.max_video_resolution,
                "target_fps": config.target_fps,
                "vram_limit_gb": config.vram_limit_gb,
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Configuration exported to: {output_path}")


# Global instance for easy access
_detector_instance: Optional[PlatformDetector] = None


def get_platform_detector(verbose: bool = True) -> PlatformDetector:
    """
    Get global PlatformDetector instance (singleton pattern).

    Args:
        verbose: Enable detailed logging

    Returns:
        Global PlatformDetector instance
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PlatformDetector(verbose=verbose)
    return _detector_instance


def detect_hardware(force_refresh: bool = False) -> HardwareInfo:
    """
    Convenience function to detect hardware.

    Args:
        force_refresh: Bypass cache and re-detect

    Returns:
        HardwareInfo object
    """
    detector = get_platform_detector()
    return detector.detect_hardware(force_refresh=force_refresh)


def get_device(prefer_gpu: bool = True, gpu_id: int = 0) -> str:
    """
    Get PyTorch device string.

    Args:
        prefer_gpu: Prefer GPU if available
        gpu_id: GPU device ID

    Returns:
        Device string ('cuda:0', 'cpu', etc.)
    """
    if not prefer_gpu:
        return "cpu"

    detector = get_platform_detector(verbose=False)
    return detector.get_device_string(gpu_id)


def get_performance_config(target_resolution: Tuple[int, int] = (1920, 1080)) -> PerformanceConfig:
    """
    Get optimal performance configuration.

    Args:
        target_resolution: Target video resolution

    Returns:
        PerformanceConfig object
    """
    detector = get_platform_detector(verbose=False)
    return detector.get_performance_config(target_resolution)


if __name__ == "__main__":
    # Demo/test mode
    print("FunGen Cross-Platform Hardware Detection\n")

    detector = PlatformDetector(verbose=True)
    hw_info = detector.detect_hardware()

    print("\nPerformance Configuration (1080p):")
    config_1080p = detector.get_performance_config((1920, 1080))
    print(f"  Batch Size: {config_1080p.batch_size}")
    print(f"  Workers: {config_1080p.num_workers}")
    print(f"  TensorRT: {config_1080p.use_tensorrt}")
    print(f"  FP16: {config_1080p.use_fp16}")
    print(f"  Optical Flow: {config_1080p.enable_optical_flow}")
    print(f"  ReID: {config_1080p.enable_reid}")
    print(f"  Target FPS: {config_1080p.target_fps}")

    print("\nPerformance Configuration (8K):")
    config_8k = detector.get_performance_config((7680, 4320))
    print(f"  Batch Size: {config_8k.batch_size}")
    print(f"  Workers: {config_8k.num_workers}")

    # Export config
    output_path = Path("hardware_config.json")
    detector.export_config(output_path)
