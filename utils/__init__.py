"""
FunGen Utilities Package

Cross-platform utilities for hardware detection, conditional imports,
and performance monitoring. Enables seamless operation on both Raspberry Pi
(CPU-only development) and RTX 3090 (GPU production).

Key Features:
- Hardware detection (CUDA, ROCm, CPU)
- Conditional GPU imports with fallbacks
- Performance profiling and FPS tracking
- Platform-specific configuration profiles
- VRAM monitoring and optimization

Target Performance:
- Raspberry Pi: 5+ FPS CPU mode
- RTX 3090: 100+ FPS GPU mode with TensorRT FP16
"""

from .conditional_imports import (  # Availability flags; Modules; Context managers; Utilities
    CUDA_AVAILABLE,
    ONNXRUNTIME_AVAILABLE,
    ONNXRUNTIME_GPU_AVAILABLE,
    OPENCV_CUDA_AVAILABLE,
    ROCM_AVAILABLE,
    TENSORRT_AVAILABLE,
    TORCH_AVAILABLE,
    GPUMemoryManager,
    ModelLoader,
    OpenCVGPU,
    autocast_context,
    get_capabilities,
    gpu_optional,
    inference_mode,
    no_grad,
    ort,
    print_capabilities,
    safe_import,
    torch,
    trt,
)
from .performance import FrameMetrics, PerformanceMonitor, PerformanceStats, Profiler, profile
from .platform_utils import (
    HardwareInfo,
    HardwareType,
    PerformanceConfig,
    PlatformDetector,
    PlatformProfile,
    detect_hardware,
    get_device,
    get_performance_config,
    get_platform_detector,
)

__version__ = "1.0.0"
__author__ = "cross-platform-dev agent"

__all__ = [
    # Platform detection
    "HardwareType",
    "PlatformProfile",
    "HardwareInfo",
    "PerformanceConfig",
    "PlatformDetector",
    "detect_hardware",
    "get_device",
    "get_performance_config",
    "get_platform_detector",
    # Conditional imports
    "TORCH_AVAILABLE",
    "CUDA_AVAILABLE",
    "ROCM_AVAILABLE",
    "TENSORRT_AVAILABLE",
    "OPENCV_CUDA_AVAILABLE",
    "ONNXRUNTIME_AVAILABLE",
    "ONNXRUNTIME_GPU_AVAILABLE",
    "torch",
    "trt",
    "ort",
    "inference_mode",
    "no_grad",
    "autocast_context",
    "GPUMemoryManager",
    "ModelLoader",
    "OpenCVGPU",
    "print_capabilities",
    "get_capabilities",
    "safe_import",
    "gpu_optional",
    # Performance monitoring
    "FrameMetrics",
    "PerformanceStats",
    "PerformanceMonitor",
    "Profiler",
    "profile",
]
