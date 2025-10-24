"""
Configuration System - Hardware profiles and settings management.

This module provides configuration management for cross-platform deployment:
- Hardware profile detection (dev_pi, prod_rtx3090, debug)
- Device-specific optimizations
- Global settings with environment variable overrides
- Runtime configuration validation

Profiles:
- dev_pi: CPU mode, minimal features, suitable for Raspberry Pi
- prod_rtx3090: GPU mode, TensorRT FP16, full features
- debug: CPU/GPU with extensive logging

Author: ml-specialist agent
Date: 2025-10-24
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# Conditional imports
try:
    import torch

    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """
    Hardware-specific configuration profile.

    Attributes:
        name: Profile name
        device: Device string (cuda:0, cpu)
        use_tensorrt: Enable TensorRT optimization
        use_fp16: Enable FP16 precision
        max_batch_size: Maximum batch size for inference
        enable_optical_flow: Enable CUDA optical flow
        enable_reid: Enable ReID for tracking
        vram_limit_gb: VRAM usage limit
        num_workers: Number of parallel workers
        log_level: Logging level
    """

    name: str
    device: str
    use_tensorrt: bool = True
    use_fp16: bool = True
    max_batch_size: int = 8
    enable_optical_flow: bool = True
    enable_reid: bool = False
    vram_limit_gb: float = 20.0
    num_workers: int = 4
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate profile settings."""
        if self.device.startswith("cuda") and not TORCH_AVAILABLE:
            logger.warning(f"Profile '{self.name}' requires CUDA but it's not available")
            self.device = "cpu"
            self.use_tensorrt = False
            self.use_fp16 = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "device": self.device,
            "use_tensorrt": self.use_tensorrt,
            "use_fp16": self.use_fp16,
            "max_batch_size": self.max_batch_size,
            "enable_optical_flow": self.enable_optical_flow,
            "enable_reid": self.enable_reid,
            "vram_limit_gb": self.vram_limit_gb,
            "num_workers": self.num_workers,
            "log_level": self.log_level,
        }


# Predefined hardware profiles
PROFILES: Dict[str, HardwareProfile] = {
    "dev_pi": HardwareProfile(
        name="dev_pi",
        device="cpu",
        use_tensorrt=False,
        use_fp16=False,
        max_batch_size=1,
        enable_optical_flow=False,  # Too slow on CPU
        enable_reid=False,  # Too slow on CPU
        vram_limit_gb=0.0,
        num_workers=1,
        log_level="DEBUG",
    ),
    "prod_rtx3090": HardwareProfile(
        name="prod_rtx3090",
        device="cuda:0",
        use_tensorrt=True,
        use_fp16=True,
        max_batch_size=8,
        enable_optical_flow=True,
        enable_reid=True,
        vram_limit_gb=20.0,  # Leave 4GB for system
        num_workers=6,
        log_level="INFO",
    ),
    "debug": HardwareProfile(
        name="debug",
        device="cuda:0" if TORCH_AVAILABLE else "cpu",
        use_tensorrt=False,  # Disable for debugging
        use_fp16=False,
        max_batch_size=1,
        enable_optical_flow=False,
        enable_reid=False,
        vram_limit_gb=20.0,
        num_workers=1,
        log_level="DEBUG",
    ),
}


class Config:
    """
    Global configuration manager with profile support.

    This class manages application-wide settings with:
    - Automatic hardware detection
    - Profile selection (manual or auto)
    - Environment variable overrides
    - Runtime validation

    Example:
        >>> config = Config.from_profile("prod_rtx3090")
        >>> print(config.device)
        'cuda:0'
        >>> config.max_batch_size = 16  # Override
    """

    def __init__(self, profile: HardwareProfile):
        """
        Initialize configuration from hardware profile.

        Args:
            profile: Hardware profile to use
        """
        self._profile = profile

        # Copy profile settings as instance attributes
        self.name = profile.name
        self.device = profile.device
        self.use_tensorrt = profile.use_tensorrt
        self.use_fp16 = profile.use_fp16
        self.max_batch_size = profile.max_batch_size
        self.enable_optical_flow = profile.enable_optical_flow
        self.enable_reid = profile.enable_reid
        self.vram_limit_gb = profile.vram_limit_gb
        self.num_workers = profile.num_workers
        self.log_level = profile.log_level

        # Additional settings
        self.model_dir = Path(os.getenv("FUNGEN_MODEL_DIR", "models/"))
        self.output_dir = Path(os.getenv("FUNGEN_OUTPUT_DIR", "output/"))
        self.cache_dir = Path(os.getenv("FUNGEN_CACHE_DIR", "cache/"))

        # Tracking settings
        self.default_tracker = os.getenv("FUNGEN_TRACKER", "hybrid")
        self.conf_threshold = float(os.getenv("FUNGEN_CONF_THRESHOLD", "0.25"))
        self.iou_threshold = float(os.getenv("FUNGEN_IOU_THRESHOLD", "0.45"))

        # Performance settings
        self.enable_profiling = os.getenv("FUNGEN_PROFILE", "false").lower() == "true"
        self.checkpoint_interval = int(os.getenv("FUNGEN_CHECKPOINT_INTERVAL", "100"))

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Configuration loaded: profile={self.name}, device={self.device}")

    @classmethod
    def from_profile(cls, profile_name: str) -> "Config":
        """
        Create configuration from named profile.

        Args:
            profile_name: Profile name (dev_pi, prod_rtx3090, debug)

        Returns:
            Config instance

        Raises:
            ValueError: If profile name not found
        """
        if profile_name not in PROFILES:
            raise ValueError(
                f"Unknown profile: {profile_name}. " f"Available: {list(PROFILES.keys())}"
            )

        profile = PROFILES[profile_name]
        return cls(profile)

    @classmethod
    def auto_detect(cls) -> "Config":
        """
        Auto-detect best hardware profile.

        Detection logic:
        1. Check FUNGEN_PROFILE environment variable
        2. If CUDA available → prod_rtx3090
        3. Otherwise → dev_pi

        Returns:
            Config instance with auto-detected profile
        """
        # Check environment variable
        env_profile = os.getenv("FUNGEN_PROFILE")
        if env_profile:
            logger.info(f"Using profile from FUNGEN_PROFILE: {env_profile}")
            return cls.from_profile(env_profile)

        # Auto-detect
        if TORCH_AVAILABLE:
            # Check GPU capabilities
            try:
                gpu_name = torch.cuda.get_device_name(0)
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                logger.info(f"Detected GPU: {gpu_name} ({total_vram:.1f} GB VRAM)")

                # Use prod profile for powerful GPUs
                if total_vram >= 20.0:
                    logger.info("Auto-selecting prod_rtx3090 profile")
                    return cls.from_profile("prod_rtx3090")
                else:
                    logger.info("GPU detected but VRAM < 20GB, using debug profile")
                    return cls.from_profile("debug")

            except Exception as e:
                logger.warning(f"GPU detection failed: {e}")

        # Fallback to CPU
        logger.info("No suitable GPU found, using dev_pi profile (CPU mode)")
        return cls.from_profile("dev_pi")

    def validate(self) -> bool:
        """
        Validate configuration settings.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Check device
        if self.device.startswith("cuda") and not TORCH_AVAILABLE:
            raise ValueError("CUDA device requested but PyTorch CUDA not available")

        # Check TensorRT
        if self.use_tensorrt and not TRT_AVAILABLE:
            logger.warning("TensorRT optimization requested but not available")
            self.use_tensorrt = False

        # Check batch size
        if self.max_batch_size < 1:
            raise ValueError(f"Invalid batch size: {self.max_batch_size}")

        # Check directories
        if not self.model_dir.exists():
            logger.warning(f"Model directory not found: {self.model_dir}")

        # Check thresholds
        if not 0.0 <= self.conf_threshold <= 1.0:
            raise ValueError(f"Invalid confidence threshold: {self.conf_threshold}")

        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"Invalid IoU threshold: {self.iou_threshold}")

        logger.info("Configuration validation passed")
        return True

    def save(self, path: Path) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Output file path
        """
        config_dict = {
            "profile": self._profile.to_dict(),
            "model_dir": str(self.model_dir),
            "output_dir": str(self.output_dir),
            "cache_dir": str(self.cache_dir),
            "default_tracker": self.default_tracker,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "enable_profiling": self.enable_profiling,
            "checkpoint_interval": self.checkpoint_interval,
        }

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "Config":
        """
        Load configuration from JSON file.

        Args:
            path: Input file path

        Returns:
            Config instance
        """
        with open(path, "r") as f:
            config_dict = json.load(f)

        # Reconstruct profile
        profile_dict = config_dict["profile"]
        profile = HardwareProfile(**profile_dict)

        # Create config
        config = cls(profile)

        # Override settings
        config.model_dir = Path(config_dict.get("model_dir", "models/"))
        config.output_dir = Path(config_dict.get("output_dir", "output/"))
        config.cache_dir = Path(config_dict.get("cache_dir", "cache/"))
        config.default_tracker = config_dict.get("default_tracker", "hybrid")
        config.conf_threshold = config_dict.get("conf_threshold", 0.25)
        config.iou_threshold = config_dict.get("iou_threshold", 0.45)
        config.enable_profiling = config_dict.get("enable_profiling", False)
        config.checkpoint_interval = config_dict.get("checkpoint_interval", 100)

        logger.info(f"Configuration loaded from {path}")
        return config

    def get_optimal_settings_for_resolution(self, width: int, height: int) -> Dict[str, Any]:
        """
        Get optimal inference settings for video resolution.

        Args:
            width: Video width in pixels
            height: Video height in pixels

        Returns:
            Dictionary with recommended settings
        """
        total_pixels = width * height

        # 1080p: 1920x1080 = 2,073,600 pixels
        # 8K: 7680x4320 = 33,177,600 pixels

        if total_pixels <= 2_100_000:  # 1080p and below
            batch_size = self.max_batch_size
            resize_factor = 1.0
            target_fps = 100
        elif total_pixels <= 8_300_000:  # 4K
            batch_size = max(1, self.max_batch_size // 2)
            resize_factor = 0.75
            target_fps = 60
        else:  # 8K and above
            batch_size = max(1, self.max_batch_size // 4)
            resize_factor = 0.5
            target_fps = 30

        return {
            "batch_size": batch_size,
            "resize_factor": resize_factor,
            "target_fps": target_fps,
            "use_fp16": self.use_fp16 and self.device.startswith("cuda"),
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config("
            f"profile={self.name}, "
            f"device={self.device}, "
            f"tensorrt={self.use_tensorrt}, "
            f"fp16={self.use_fp16}, "
            f"batch={self.max_batch_size}"
            f")"
        )


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance (singleton).

    Returns:
        Global Config instance
    """
    global _global_config

    if _global_config is None:
        _global_config = Config.auto_detect()
        _global_config.validate()

    return _global_config


def set_config(config: Config) -> None:
    """
    Set global configuration instance.

    Args:
        config: Config instance to set as global
    """
    global _global_config
    config.validate()
    _global_config = config


def reset_config() -> None:
    """Reset global configuration to None."""
    global _global_config
    _global_config = None


if __name__ == "__main__":
    # Test configuration
    print("Available profiles:")
    for name, profile in PROFILES.items():
        print(f"  {name}: {profile}")

    print("\nAuto-detected configuration:")
    config = Config.auto_detect()
    print(config)
    print(f"  Model dir: {config.model_dir}")
    print(f"  Device: {config.device}")
    print(f"  TensorRT: {config.use_tensorrt}")
    print(f"  FP16: {config.use_fp16}")

    # Test resolution-specific settings
    print("\nResolution-specific settings:")
    for res in [(1920, 1080), (3840, 2160), (7680, 4320)]:
        settings = config.get_optimal_settings_for_resolution(*res)
        print(
            f"  {res[0]}x{res[1]}: batch={settings['batch_size']}, fps_target={settings['target_fps']}"
        )
