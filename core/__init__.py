"""Core ML Infrastructure - FunGen Rewrite.

This package provides high-performance ML infrastructure for YOLO-based
video tracking with TensorRT optimization.

Modules:
- model_manager: YOLO loading, TensorRT optimization, batch inference
- tensorrt_converter: FP16/INT8 model conversion and benchmarking
- config: Hardware profiles and configuration management

Author: ml-specialist agent
Date: 2025-10-24
"""

from .config import PROFILES, Config, HardwareProfile, get_config, reset_config, set_config
from .model_manager import (
    TORCH_AVAILABLE,
    TRT_AVAILABLE,
    ULTRALYTICS_AVAILABLE,
    Detection,
    ModelInfo,
    ModelManager,
)
from .tensorrt_converter import TensorRTConverter, optimize_model_for_rtx3090

__all__ = [
    # ModelManager
    "ModelManager",
    "Detection",
    "ModelInfo",
    "TORCH_AVAILABLE",
    "ULTRALYTICS_AVAILABLE",
    "TRT_AVAILABLE",
    # Config
    "Config",
    "HardwareProfile",
    "PROFILES",
    "get_config",
    "set_config",
    "reset_config",
    # TensorRT
    "TensorRTConverter",
    "optimize_model_for_rtx3090",
]

__version__ = "1.0.0"
__author__ = "ml-specialist agent"
