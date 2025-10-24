"""
Unit tests for conditional_imports.py

Tests conditional GPU imports, fallbacks, and capability detection.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.conditional_imports import (
    CUDA_AVAILABLE,
    ONNXRUNTIME_AVAILABLE,
    OPENCV_CUDA_AVAILABLE,
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
    safe_import,
)


class TestCapabilityFlags(unittest.TestCase):
    """Tests for capability detection flags."""

    def test_flags_are_boolean(self):
        """Test that all flags are boolean."""
        self.assertIsInstance(TORCH_AVAILABLE, bool)
        self.assertIsInstance(CUDA_AVAILABLE, bool)
        self.assertIsInstance(TENSORRT_AVAILABLE, bool)
        self.assertIsInstance(OPENCV_CUDA_AVAILABLE, bool)
        self.assertIsInstance(ONNXRUNTIME_AVAILABLE, bool)

    def test_cuda_requires_torch(self):
        """Test that CUDA availability requires PyTorch."""
        if CUDA_AVAILABLE:
            self.assertTrue(TORCH_AVAILABLE, "CUDA requires PyTorch")

    def test_get_capabilities_dict(self):
        """Test get_capabilities returns valid dictionary."""
        caps = get_capabilities()

        self.assertIsInstance(caps, dict)
        self.assertIn("torch_available", caps)
        self.assertIn("cuda_available", caps)
        self.assertIn("tensorrt_available", caps)
        self.assertIn("optimal_provider", caps)


class TestContextManagers(unittest.TestCase):
    """Tests for context managers."""

    def test_inference_mode(self):
        """Test inference_mode context manager."""
        with inference_mode():
            # Should not raise an exception
            pass

    def test_no_grad(self):
        """Test no_grad context manager."""
        with no_grad():
            # Should not raise an exception
            pass

    def test_autocast_context(self):
        """Test autocast_context context manager."""
        with autocast_context():
            # Should not raise an exception
            pass

        with autocast_context(device_type="cpu"):
            # Should not raise an exception
            pass


class TestGPUMemoryManager(unittest.TestCase):
    """Tests for GPUMemoryManager."""

    def test_get_memory_info(self):
        """Test get_memory_info returns tuple."""
        used, total = GPUMemoryManager.get_memory_info()

        self.assertIsInstance(used, float)
        self.assertIsInstance(total, float)
        self.assertGreaterEqual(used, 0.0)
        self.assertGreaterEqual(total, 0.0)

        if CUDA_AVAILABLE:
            # GPU should have some memory
            self.assertGreater(total, 0.0)
        else:
            # CPU should return zeros
            self.assertEqual(used, 0.0)
            self.assertEqual(total, 0.0)

    def test_empty_cache(self):
        """Test empty_cache doesn't raise exception."""
        # Should work regardless of GPU availability
        GPUMemoryManager.empty_cache()

    def test_synchronize(self):
        """Test synchronize doesn't raise exception."""
        # Should work regardless of GPU availability
        GPUMemoryManager.synchronize()

    def test_set_memory_fraction(self):
        """Test set_memory_fraction doesn't raise exception."""
        # Should work regardless of GPU availability
        GPUMemoryManager.set_memory_fraction(0.8)


class TestModelLoader(unittest.TestCase):
    """Tests for ModelLoader."""

    def test_get_optimal_provider(self):
        """Test get_optimal_provider returns valid string."""
        provider = ModelLoader.get_optimal_provider()

        self.assertIsInstance(provider, str)
        self.assertIn(provider, ["tensorrt", "cuda", "onnxruntime-gpu", "onnxruntime-cpu", "cpu"])

    def test_get_onnx_providers(self):
        """Test get_onnx_providers returns list."""
        providers = ModelLoader.get_onnx_providers()

        self.assertIsInstance(providers, list)

        if ONNXRUNTIME_AVAILABLE:
            # Should at least have CPUExecutionProvider
            self.assertGreater(len(providers), 0)
            self.assertIn("CPUExecutionProvider", providers)
        else:
            # Should return empty list
            self.assertEqual(len(providers), 0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required")
    def test_load_pytorch_model_validation(self):
        """Test PyTorch model loading parameter validation."""
        # Test with non-existent file should raise appropriate error
        with self.assertRaises(Exception):
            ModelLoader.load_pytorch_model("/nonexistent/model.pt")


class TestOpenCVGPU(unittest.TestCase):
    """Tests for OpenCVGPU utilities."""

    def test_is_available(self):
        """Test is_available returns boolean."""
        available = OpenCVGPU.is_available()
        self.assertIsInstance(available, bool)

    def test_create_optical_flow(self):
        """Test create_optical_flow returns object or None."""
        flow = OpenCVGPU.create_optical_flow()

        if OPENCV_CUDA_AVAILABLE:
            # Should return flow object
            self.assertIsNotNone(flow)
        else:
            # Should return None gracefully
            self.assertIsNone(flow)

    def test_create_gpu_mat(self):
        """Test create_gpu_mat returns object or None."""
        gpu_mat = OpenCVGPU.create_gpu_mat()

        if OPENCV_CUDA_AVAILABLE:
            # Should return GpuMat object
            self.assertIsNotNone(gpu_mat)
        else:
            # Should return None gracefully
            self.assertIsNone(gpu_mat)


class TestSafeImport(unittest.TestCase):
    """Tests for safe_import utility."""

    def test_safe_import_existing_module(self):
        """Test safe import with existing module."""
        module, success = safe_import("sys")
        self.assertTrue(success)
        self.assertIsNotNone(module)

    def test_safe_import_nonexistent_module(self):
        """Test safe import with non-existent module."""
        module, success = safe_import("nonexistent_module_xyz")
        self.assertFalse(success)
        self.assertIsNotNone(module)  # Should return MagicMock

    def test_safe_import_with_fallback(self):
        """Test safe import with custom fallback."""
        fallback = "custom_fallback"
        module, success = safe_import("nonexistent_module", fallback=fallback)
        self.assertFalse(success)
        self.assertEqual(module, fallback)


class TestGPUOptionalDecorator(unittest.TestCase):
    """Tests for gpu_optional decorator."""

    def test_gpu_optional_with_cuda(self):
        """Test gpu_optional decorator when CUDA available."""

        def cpu_func():
            return "cpu"

        @gpu_optional(cpu_fallback=cpu_func)
        def gpu_func():
            return "gpu"

        result = gpu_func()

        if CUDA_AVAILABLE:
            self.assertEqual(result, "gpu")
        else:
            self.assertEqual(result, "cpu")

    def test_gpu_optional_without_fallback(self):
        """Test gpu_optional decorator without CPU fallback."""

        @gpu_optional()
        def gpu_only_func():
            return "gpu"

        if CUDA_AVAILABLE:
            result = gpu_only_func()
            self.assertEqual(result, "gpu")
        else:
            with self.assertRaises(RuntimeError):
                gpu_only_func()

    def test_gpu_optional_with_args(self):
        """Test gpu_optional decorator with function arguments."""

        def cpu_func(x, y):
            return x + y

        @gpu_optional(cpu_fallback=cpu_func)
        def gpu_func(x, y):
            return x * y

        if CUDA_AVAILABLE:
            result = gpu_func(5, 3)
            self.assertEqual(result, 15)
        else:
            result = gpu_func(5, 3)
            self.assertEqual(result, 8)


class TestMockObjects(unittest.TestCase):
    """Tests for mock objects in CPU-only mode."""

    @unittest.skipIf(TORCH_AVAILABLE, "Test for CPU-only mode")
    def test_torch_mock(self):
        """Test torch mock object in CPU-only mode."""
        from utils.conditional_imports import torch

        # Mock should have cuda attribute
        self.assertTrue(hasattr(torch, "cuda"))

        # cuda.is_available should return False
        self.assertFalse(torch.cuda.is_available())

    @unittest.skipIf(TENSORRT_AVAILABLE, "Test for non-TensorRT mode")
    def test_tensorrt_mock(self):
        """Test TensorRT mock object when not available."""
        from utils.conditional_imports import trt

        # Should be a mock object
        self.assertIsNotNone(trt)


class TestIntegration(unittest.TestCase):
    """Integration tests for conditional imports."""

    def test_full_import_chain(self):
        """Test importing all utilities together."""
        try:
            from utils.conditional_imports import (
                CUDA_AVAILABLE,
                TORCH_AVAILABLE,
                GPUMemoryManager,
                ModelLoader,
                OpenCVGPU,
                autocast_context,
                inference_mode,
                no_grad,
            )

            # Should not raise any exceptions
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Import chain failed: {e}")

    def test_context_managers_work_together(self):
        """Test using multiple context managers together."""
        try:
            with inference_mode():
                with no_grad():
                    with autocast_context():
                        # Should not raise exceptions
                        pass
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Context managers failed: {e}")

    def test_memory_operations_sequence(self):
        """Test sequence of memory operations."""
        try:
            GPUMemoryManager.empty_cache()
            used, total = GPUMemoryManager.get_memory_info()
            GPUMemoryManager.synchronize()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Memory operations failed: {e}")


if __name__ == "__main__":
    unittest.main()
