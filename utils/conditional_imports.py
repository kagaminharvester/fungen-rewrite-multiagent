"""
Conditional GPU imports with graceful fallbacks for cross-platform compatibility.

This module provides safe imports of GPU-dependent libraries (PyTorch, TensorRT,
CUDA, etc.) with automatic fallbacks for CPU-only environments (Raspberry Pi).

Design Philosophy:
- Never crash on import - always provide fallback
- Detect capabilities early and cache results
- Provide mock objects for missing dependencies
- Enable seamless CPU/GPU code sharing
"""

import sys
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Optional, Tuple
from unittest.mock import MagicMock

# =============================================================================
# Import Availability Flags
# =============================================================================

TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
ROCM_AVAILABLE = False
TENSORRT_AVAILABLE = False
OPENCV_CUDA_AVAILABLE = False
ONNXRUNTIME_AVAILABLE = False
ONNXRUNTIME_GPU_AVAILABLE = False

# =============================================================================
# PyTorch Imports
# =============================================================================

try:
    import torch

    TORCH_AVAILABLE = True

    # Check CUDA availability
    try:
        CUDA_AVAILABLE = torch.cuda.is_available()
    except Exception:
        CUDA_AVAILABLE = False

    # Check ROCm availability
    try:
        ROCM_AVAILABLE = hasattr(torch, "hip") and torch.hip.is_available()
    except Exception:
        ROCM_AVAILABLE = False

except ImportError:
    # Create mock torch module for CPU-only environments
    torch = MagicMock()
    torch.cuda.is_available = lambda: False
    torch.Tensor = object
    torch.nn.Module = object
    warnings.warn(
        "PyTorch not available. Using CPU fallback mode. " "GPU features disabled.", ImportWarning
    )

# =============================================================================
# TensorRT Imports
# =============================================================================

try:
    import tensorrt as trt

    TENSORRT_AVAILABLE = True
except ImportError:
    trt = MagicMock()
    TENSORRT_AVAILABLE = False
    if CUDA_AVAILABLE:
        warnings.warn(
            "TensorRT not available. GPU optimization disabled. "
            "Install with: pip install nvidia-tensorrt",
            ImportWarning,
        )

# =============================================================================
# OpenCV CUDA Imports
# =============================================================================

try:
    import cv2

    # Check if OpenCV was built with CUDA support
    if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        OPENCV_CUDA_AVAILABLE = True
    else:
        OPENCV_CUDA_AVAILABLE = False
except Exception:
    OPENCV_CUDA_AVAILABLE = False

# =============================================================================
# ONNX Runtime Imports
# =============================================================================

try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True

    # Check for GPU provider
    available_providers = ort.get_available_providers()
    ONNXRUNTIME_GPU_AVAILABLE = (
        "CUDAExecutionProvider" in available_providers
        or "TensorrtExecutionProvider" in available_providers
    )
except ImportError:
    ort = MagicMock()
    ONNXRUNTIME_AVAILABLE = False
    warnings.warn(
        "ONNX Runtime not available. Model inference may be slower. "
        "Install with: pip install onnxruntime or onnxruntime-gpu",
        ImportWarning,
    )

# =============================================================================
# Conditional Context Managers
# =============================================================================

if TORCH_AVAILABLE and CUDA_AVAILABLE:
    # Use real PyTorch inference mode
    from torch import inference_mode, no_grad

    def autocast_context(device_type: str = "cuda", dtype=None):
        """Context manager for automatic mixed precision."""
        if dtype is None:
            dtype = torch.float16 if device_type == "cuda" else torch.float32
        return torch.autocast(device_type=device_type, dtype=dtype)

else:
    # Provide no-op context managers for CPU-only mode
    @contextmanager
    def inference_mode():
        """No-op inference mode for CPU."""
        yield

    @contextmanager
    def no_grad():
        """No-op gradient disabling for CPU."""
        yield

    @contextmanager
    def autocast_context(device_type: str = "cpu", dtype=None):
        """No-op autocast for CPU."""
        yield


# =============================================================================
# GPU Memory Management
# =============================================================================


class GPUMemoryManager:
    """Manages GPU memory allocation and monitoring."""

    @staticmethod
    def get_memory_info() -> Tuple[float, float]:
        """
        Get GPU memory usage.

        Returns:
            Tuple of (used_memory_gb, total_memory_gb)
        """
        if not CUDA_AVAILABLE:
            return (0.0, 0.0)

        try:
            import torch

            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            return (allocated, reserved)
        except Exception:
            return (0.0, 0.0)

    @staticmethod
    def empty_cache() -> None:
        """Clear GPU cache."""
        if CUDA_AVAILABLE:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    @staticmethod
    def synchronize() -> None:
        """Synchronize CUDA operations."""
        if CUDA_AVAILABLE:
            try:
                import torch

                torch.cuda.synchronize()
            except Exception:
                pass

    @staticmethod
    def set_memory_fraction(fraction: float = 0.9) -> None:
        """
        Set maximum GPU memory fraction.

        Args:
            fraction: Fraction of GPU memory to use (0.0-1.0)
        """
        if CUDA_AVAILABLE:
            try:
                import torch

                torch.cuda.set_per_process_memory_fraction(fraction)
            except Exception:
                pass


# =============================================================================
# Model Loading Utilities
# =============================================================================


class ModelLoader:
    """Handles loading ML models with appropriate backend."""

    @staticmethod
    def get_optimal_provider() -> str:
        """
        Get optimal execution provider for current hardware.

        Returns:
            Provider name ('cuda', 'tensorrt', 'onnxruntime-gpu', 'onnxruntime-cpu', 'cpu')
        """
        if TENSORRT_AVAILABLE and CUDA_AVAILABLE:
            return "tensorrt"
        elif CUDA_AVAILABLE:
            return "cuda"
        elif ONNXRUNTIME_GPU_AVAILABLE:
            return "onnxruntime-gpu"
        elif ONNXRUNTIME_AVAILABLE:
            return "onnxruntime-cpu"
        else:
            return "cpu"

    @staticmethod
    def get_onnx_providers() -> list:
        """
        Get prioritized list of ONNX Runtime providers.

        Returns:
            List of provider names in priority order
        """
        if not ONNXRUNTIME_AVAILABLE:
            return []

        providers = []

        # Prefer TensorRT if available
        if "TensorrtExecutionProvider" in ort.get_available_providers():
            providers.append("TensorrtExecutionProvider")

        # Then CUDA
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")

        # Always include CPU as fallback
        providers.append("CPUExecutionProvider")

        return providers

    @staticmethod
    def load_pytorch_model(
        model_path: str, device: Optional[str] = None, fp16: bool = False
    ) -> Any:
        """
        Load PyTorch model with appropriate device.

        Args:
            model_path: Path to model file
            device: Target device ('cuda', 'cpu', or None for auto)
            fp16: Use FP16 precision

        Returns:
            Loaded model
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        import torch

        # Auto-detect device
        if device is None:
            device = "cuda" if CUDA_AVAILABLE else "cpu"

        # Load model
        model = torch.load(model_path, map_location=device)

        # Convert to FP16 if requested and supported
        if fp16 and device == "cuda":
            model = model.half()

        model.eval()
        return model

    @staticmethod
    def load_onnx_model(model_path: str, providers: Optional[list] = None) -> Any:
        """
        Load ONNX model with optimal providers.

        Args:
            model_path: Path to ONNX model
            providers: List of providers (None for auto)

        Returns:
            ONNX Runtime InferenceSession
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")

        if providers is None:
            providers = ModelLoader.get_onnx_providers()

        import onnxruntime as ort

        session = ort.InferenceSession(model_path, providers=providers)

        return session


# =============================================================================
# OpenCV CUDA Utilities
# =============================================================================


class OpenCVGPU:
    """OpenCV CUDA acceleration utilities."""

    @staticmethod
    def create_optical_flow():
        """
        Create GPU-accelerated optical flow object.

        Returns:
            cv2.cuda optical flow object or None if unavailable
        """
        if not OPENCV_CUDA_AVAILABLE:
            return None

        try:
            import cv2

            flow = cv2.cuda.FarnebackOpticalFlow_create(
                numLevels=3,
                pyrScale=0.5,
                fastPyramids=True,
                winSize=15,
                numIters=3,
                polyN=5,
                polySigma=1.2,
            )
            return flow
        except Exception as e:
            warnings.warn(f"Failed to create GPU optical flow: {e}")
            return None

    @staticmethod
    def create_gpu_mat(image=None):
        """
        Create GPU matrix (GpuMat).

        Args:
            image: Optional numpy array to upload

        Returns:
            cv2.cuda.GpuMat or None
        """
        if not OPENCV_CUDA_AVAILABLE:
            return None

        try:
            import cv2

            gpu_mat = cv2.cuda.GpuMat()
            if image is not None:
                gpu_mat.upload(image)
            return gpu_mat
        except Exception:
            return None

    @staticmethod
    def is_available() -> bool:
        """Check if OpenCV CUDA is available."""
        return OPENCV_CUDA_AVAILABLE


# =============================================================================
# Capability Summary
# =============================================================================


def print_capabilities(detailed: bool = False) -> None:
    """
    Print available GPU capabilities.

    Args:
        detailed: Show detailed information
    """
    print("\n" + "=" * 60)
    print("GPU CAPABILITIES SUMMARY")
    print("=" * 60)

    capabilities = {
        "PyTorch": TORCH_AVAILABLE,
        "CUDA (PyTorch)": CUDA_AVAILABLE,
        "ROCm": ROCM_AVAILABLE,
        "TensorRT": TENSORRT_AVAILABLE,
        "OpenCV CUDA": OPENCV_CUDA_AVAILABLE,
        "ONNX Runtime": ONNXRUNTIME_AVAILABLE,
        "ONNX Runtime GPU": ONNXRUNTIME_GPU_AVAILABLE,
    }

    for name, available in capabilities.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"{name:20s}: {status}")

    if detailed:
        print("\nOptimal Provider:", ModelLoader.get_optimal_provider())

        if ONNXRUNTIME_AVAILABLE:
            print("ONNX Providers:", ", ".join(ModelLoader.get_onnx_providers()))

        if CUDA_AVAILABLE:
            used, total = GPUMemoryManager.get_memory_info()
            print(f"GPU Memory: {used:.2f}GB used, {total:.2f}GB reserved")

    print("=" * 60 + "\n")


def get_capabilities() -> dict:
    """
    Get capabilities as dictionary.

    Returns:
        Dictionary of capability flags
    """
    return {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": CUDA_AVAILABLE,
        "rocm_available": ROCM_AVAILABLE,
        "tensorrt_available": TENSORRT_AVAILABLE,
        "opencv_cuda_available": OPENCV_CUDA_AVAILABLE,
        "onnxruntime_available": ONNXRUNTIME_AVAILABLE,
        "onnxruntime_gpu_available": ONNXRUNTIME_GPU_AVAILABLE,
        "optimal_provider": ModelLoader.get_optimal_provider(),
    }


# =============================================================================
# Safe Import Wrapper
# =============================================================================


def safe_import(module_name: str, fallback: Any = None) -> Tuple[Any, bool]:
    """
    Safely import a module with fallback.

    Args:
        module_name: Name of module to import
        fallback: Fallback object if import fails

    Returns:
        Tuple of (module, success_flag)
    """
    try:
        module = __import__(module_name)
        return module, True
    except ImportError:
        if fallback is not None:
            return fallback, False
        return MagicMock(), False


# =============================================================================
# Decorator for GPU-Optional Functions
# =============================================================================


def gpu_optional(cpu_fallback: Optional[Callable] = None):
    """
    Decorator to provide CPU fallback for GPU functions.

    Args:
        cpu_fallback: Function to call when GPU not available

    Example:
        @gpu_optional(cpu_fallback=cpu_version)
        def gpu_process(data):
            # GPU implementation
            pass
    """

    def decorator(gpu_func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if CUDA_AVAILABLE:
                return gpu_func(*args, **kwargs)
            elif cpu_fallback is not None:
                return cpu_fallback(*args, **kwargs)
            else:
                raise RuntimeError(
                    f"GPU not available and no CPU fallback provided for {gpu_func.__name__}"
                )

        return wrapper

    return decorator


# =============================================================================
# Module Initialization
# =============================================================================

# Print warning if running on CPU-only
if not CUDA_AVAILABLE and not ROCM_AVAILABLE:
    warnings.warn(
        "Running in CPU-only mode. GPU acceleration disabled. "
        "Performance will be significantly reduced. "
        "Expected FPS: 5-10 (Pi) vs 100+ (RTX 3090)",
        RuntimeWarning,
    )

# Export all flags and utilities
__all__ = [
    # Availability flags
    "TORCH_AVAILABLE",
    "CUDA_AVAILABLE",
    "ROCM_AVAILABLE",
    "TENSORRT_AVAILABLE",
    "OPENCV_CUDA_AVAILABLE",
    "ONNXRUNTIME_AVAILABLE",
    "ONNXRUNTIME_GPU_AVAILABLE",
    # Modules
    "torch",
    "trt",
    "ort",
    # Context managers
    "inference_mode",
    "no_grad",
    "autocast_context",
    # Utilities
    "GPUMemoryManager",
    "ModelLoader",
    "OpenCVGPU",
    "print_capabilities",
    "get_capabilities",
    "safe_import",
    "gpu_optional",
]

if __name__ == "__main__":
    # Demo mode
    print("FunGen Conditional Imports - Capability Detection\n")
    print_capabilities(detailed=True)

    # Test GPU memory manager
    if CUDA_AVAILABLE:
        print("\nTesting GPU Memory Manager...")
        GPUMemoryManager.empty_cache()
        used, reserved = GPUMemoryManager.get_memory_info()
        print(f"Memory after cache clear: {used:.2f}GB used, {reserved:.2f}GB reserved")

    # Test model loader
    print("\nOptimal inference provider:", ModelLoader.get_optimal_provider())
    if ONNXRUNTIME_AVAILABLE:
        print("ONNX providers:", ModelLoader.get_onnx_providers())
