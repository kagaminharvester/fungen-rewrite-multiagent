"""
Unit tests for ModelManager - YOLO model loading and TensorRT optimization.

Test coverage:
- Model loading with auto-format detection
- TensorRT conversion and optimization
- Batch inference with dynamic sizing
- VRAM monitoring and tracking
- Device selection and fallbacks
- Performance statistics
- Error handling and edge cases

Author: ml-specialist agent
Date: 2025-10-24
"""

import json

# Import module under test
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_manager import (
    TORCH_AVAILABLE,
    TRT_AVAILABLE,
    ULTRALYTICS_AVAILABLE,
    Detection,
    ModelInfo,
    ModelManager,
)


@pytest.fixture
def temp_model_dir():
    """Create temporary model directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        yield model_dir


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing."""
    model = Mock()
    model.to = Mock(return_value=model)
    model.predict = Mock()
    model.export = Mock()

    # Mock results
    mock_result = Mock()
    mock_result.boxes = Mock()
    mock_result.boxes.xyxy = Mock()
    mock_result.boxes.xyxy.cpu = Mock()
    mock_result.boxes.xyxy.cpu().numpy = Mock(return_value=np.array([[10, 20, 100, 200]]))
    mock_result.boxes.conf = Mock()
    mock_result.boxes.conf.cpu = Mock()
    mock_result.boxes.conf.cpu().numpy = Mock(return_value=np.array([0.95]))
    mock_result.boxes.cls = Mock()
    mock_result.boxes.cls.cpu = Mock()
    mock_result.boxes.cls.cpu().numpy = Mock(return_value=np.array([0]))
    mock_result.names = {0: "test_class"}

    model.predict.return_value = [mock_result]

    return model


class TestDetection:
    """Test Detection dataclass."""

    def test_detection_creation(self):
        """Test creating Detection object."""
        det = Detection(bbox=(10, 20, 100, 200), confidence=0.95, class_id=0, class_name="person")

        assert det.bbox == (10, 20, 100, 200)
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == "person"

    def test_detection_bbox_coordinates(self):
        """Test bbox coordinate format (x1, y1, x2, y2)."""
        det = Detection(bbox=(50, 60, 150, 160), confidence=0.8, class_id=1, class_name="hand")

        x1, y1, x2, y2 = det.bbox
        assert x2 > x1  # x2 should be greater than x1
        assert y2 > y1  # y2 should be greater than y1


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating ModelInfo object."""
        info = ModelInfo(
            path=Path("models/yolo11n.engine"),
            format="engine",
            precision="fp16",
            input_size=(640, 640),
            vram_mb=2048,
            device="cuda:0",
        )

        assert info.format == "engine"
        assert info.precision == "fp16"
        assert info.input_size == (640, 640)
        assert info.vram_mb == 2048


class TestModelManagerInitialization:
    """Test ModelManager initialization."""

    def test_init_with_valid_directory(self, temp_model_dir):
        """Test initialization with valid model directory."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")

        assert manager.model_dir == temp_model_dir
        assert manager.device == "cpu"
        assert manager.model is None
        assert manager.max_batch_size == 8

    def test_init_with_invalid_directory(self):
        """Test initialization with non-existent directory."""
        with pytest.raises(FileNotFoundError):
            ModelManager(model_dir="/nonexistent/path")

    def test_init_auto_device_cpu(self):
        """Test auto device selection falls back to CPU."""
        with patch("core.model_manager.TORCH_AVAILABLE", False):
            manager = ModelManager(model_dir=".", device="auto")
            assert manager.device == "cpu"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires PyTorch CUDA")
    def test_init_auto_device_cuda(self):
        """Test auto device selection with CUDA available."""
        with patch("core.model_manager.TORCH_AVAILABLE", True):
            with patch("core.model_manager.torch.cuda.is_available", return_value=True):
                with patch(
                    "core.model_manager.torch.cuda.get_device_name", return_value="RTX 3090"
                ):
                    manager = ModelManager(model_dir=".", device="auto")
                    assert manager.device == "cuda:0"

    def test_init_cuda_without_torch_raises_error(self):
        """Test CUDA device request without PyTorch raises error."""
        with patch("core.model_manager.TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="PyTorch not available"):
                ModelManager(model_dir=".", device="cuda")

    def test_init_custom_batch_size(self, temp_model_dir):
        """Test initialization with custom batch size."""
        manager = ModelManager(model_dir=temp_model_dir, max_batch_size=16)
        assert manager.max_batch_size == 16


class TestModelFinding:
    """Test model file detection."""

    def test_find_engine_model(self, temp_model_dir):
        """Test finding .engine model (highest priority)."""
        # Create mock files
        (temp_model_dir / "yolo11n.engine").touch()
        (temp_model_dir / "yolo11n.pt").touch()

        manager = ModelManager(model_dir=temp_model_dir, device="cpu")
        found = manager.find_model("yolo11n")

        assert found == temp_model_dir / "yolo11n.engine"

    def test_find_onnx_model(self, temp_model_dir):
        """Test finding .onnx model (second priority)."""
        (temp_model_dir / "yolo11n.onnx").touch()
        (temp_model_dir / "yolo11n.pt").touch()

        manager = ModelManager(model_dir=temp_model_dir, device="cpu")
        found = manager.find_model("yolo11n")

        assert found == temp_model_dir / "yolo11n.onnx"

    def test_find_pt_model(self, temp_model_dir):
        """Test finding .pt model (lowest priority)."""
        (temp_model_dir / "yolo11n.pt").touch()

        manager = ModelManager(model_dir=temp_model_dir, device="cpu")
        found = manager.find_model("yolo11n")

        assert found == temp_model_dir / "yolo11n.pt"

    def test_find_nonexistent_model(self, temp_model_dir):
        """Test finding model that doesn't exist returns None."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")
        found = manager.find_model("nonexistent")

        assert found is None


class TestModelLoading:
    """Test model loading."""

    @pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="Requires Ultralytics")
    def test_load_model_success(self, temp_model_dir, mock_yolo_model):
        """Test successful model loading."""
        # Create mock model file
        model_path = temp_model_dir / "yolo11n.pt"
        model_path.touch()

        manager = ModelManager(model_dir=temp_model_dir, device="cpu", warmup=False)

        with patch("core.model_manager.YOLO", return_value=mock_yolo_model):
            success = manager.load_model("yolo11n", optimize=False)

            assert success is True
            assert manager.model is not None
            assert manager.loaded_model_name == "yolo11n"

    def test_load_model_without_ultralytics_fails(self, temp_model_dir):
        """Test model loading fails without Ultralytics."""
        model_path = temp_model_dir / "yolo11n.pt"
        model_path.touch()

        manager = ModelManager(model_dir=temp_model_dir, device="cpu")

        with patch("core.model_manager.ULTRALYTICS_AVAILABLE", False):
            success = manager.load_model("yolo11n")
            assert success is False

    def test_load_nonexistent_model(self, temp_model_dir):
        """Test loading non-existent model returns False."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")
        success = manager.load_model("nonexistent")
        assert success is False


class TestBatchInference:
    """Test batch inference."""

    @pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="Requires Ultralytics")
    def test_predict_batch_success(self, temp_model_dir, mock_yolo_model):
        """Test successful batch inference."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu", warmup=False)
        manager.model = mock_yolo_model
        manager.model_info = ModelInfo(
            path=Path("test.pt"),
            format="pt",
            precision="fp32",
            input_size=(640, 640),
            vram_mb=1024,
            device="cpu",
        )

        # Create test frames
        frames = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(4)]

        # Run inference
        detections = manager.predict_batch(frames)

        assert len(detections) == 4  # One result per frame
        assert isinstance(detections[0], list)
        assert len(detections[0]) > 0  # At least one detection

        det = detections[0][0]
        assert isinstance(det, Detection)
        assert det.confidence > 0.0

    def test_predict_batch_without_model_raises_error(self, temp_model_dir):
        """Test batch inference without loaded model raises error."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")

        frames = [np.zeros((640, 640, 3), dtype=np.uint8)]

        with pytest.raises(RuntimeError, match="Model not loaded"):
            manager.predict_batch(frames)

    def test_predict_batch_empty_frames(self, temp_model_dir, mock_yolo_model):
        """Test batch inference with empty frame list."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")
        manager.model = mock_yolo_model

        detections = manager.predict_batch([])
        assert detections == []

    def test_predict_batch_exceeds_max_size(self, temp_model_dir, mock_yolo_model):
        """Test batch size limiting."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu", max_batch_size=4)
        manager.model = mock_yolo_model
        manager.model_info = ModelInfo(
            path=Path("test.pt"),
            format="pt",
            precision="fp32",
            input_size=(640, 640),
            vram_mb=1024,
            device="cpu",
        )

        # Create 10 frames (exceeds max_batch_size of 4)
        frames = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(10)]

        with patch.object(manager.model, "predict") as mock_predict:
            # Setup mock to return correct number of results
            mock_predict.return_value = [Mock(boxes=None) for _ in range(4)]

            detections = manager.predict_batch(frames)

            # Should only process first 4 frames
            assert len(detections) == 4


class TestVRAMMonitoring:
    """Test VRAM monitoring."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires PyTorch CUDA")
    def test_get_vram_usage(self, temp_model_dir):
        """Test VRAM usage reporting."""
        with patch("core.model_manager.torch.cuda.is_available", return_value=True):
            with patch("core.model_manager.torch.cuda.memory_allocated", return_value=2 * 1024**3):
                manager = ModelManager(model_dir=temp_model_dir, device="cuda:0")
                vram = manager.get_vram_usage()
                assert vram == 2.0  # 2 GB

    def test_get_vram_usage_cpu_mode(self, temp_model_dir):
        """Test VRAM usage returns 0 in CPU mode."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")
        vram = manager.get_vram_usage()
        assert vram == 0.0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires PyTorch CUDA")
    def test_get_vram_peak(self, temp_model_dir):
        """Test peak VRAM reporting."""
        with patch("core.model_manager.torch.cuda.is_available", return_value=True):
            with patch(
                "core.model_manager.torch.cuda.max_memory_allocated", return_value=5 * 1024**3
            ):
                manager = ModelManager(model_dir=temp_model_dir, device="cuda:0")
                peak = manager.get_vram_peak()
                assert peak == 5.0  # 5 GB


class TestDynamicBatchSizing:
    """Test dynamic batch size calculation."""

    def test_get_optimal_batch_size_cpu(self, temp_model_dir):
        """Test optimal batch size for CPU mode."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")
        batch_size = manager.get_optimal_batch_size()
        assert batch_size == 1  # CPU should use batch size 1

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires PyTorch CUDA")
    def test_get_optimal_batch_size_with_available_vram(self, temp_model_dir):
        """Test optimal batch size calculation with available VRAM."""
        manager = ModelManager(model_dir=temp_model_dir, device="cuda:0", max_batch_size=16)
        manager.model_info = ModelInfo(
            path=Path("test.engine"),
            format="engine",
            precision="fp16",
            input_size=(640, 640),
            vram_mb=500,  # 500 MB per image
            device="cuda:0",
        )

        # Test with 10 GB available
        batch_size = manager.get_optimal_batch_size(available_vram_gb=10.0)

        # 10 GB / 0.5 GB per image * 0.8 safety = 16 images
        assert batch_size == 16  # Clamped to max_batch_size

    def test_get_optimal_batch_size_limited_vram(self, temp_model_dir):
        """Test optimal batch size with limited VRAM."""
        manager = ModelManager(model_dir=temp_model_dir, device="cuda:0", max_batch_size=16)
        manager.model_info = ModelInfo(
            path=Path("test.engine"),
            format="engine",
            precision="fp16",
            input_size=(640, 640),
            vram_mb=2000,  # 2 GB per image
            device="cuda:0",
        )

        # Test with 4 GB available
        batch_size = manager.get_optimal_batch_size(available_vram_gb=4.0)

        # 4 GB / 2 GB per image * 0.8 safety = 1.6 → 1 image
        assert batch_size == 1


class TestPerformanceStats:
    """Test performance statistics."""

    def test_get_performance_stats_no_data(self, temp_model_dir):
        """Test performance stats with no inference data."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")
        stats = manager.get_performance_stats()

        assert stats["avg_fps"] == 0.0
        assert stats["avg_latency_ms"] == 0.0
        assert "vram_usage_gb" in stats

    def test_get_performance_stats_with_data(self, temp_model_dir):
        """Test performance stats with inference data."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")

        # Simulate inference data
        manager.inference_times = [0.1, 0.15, 0.12]  # seconds
        manager.batch_sizes = [4, 4, 4]

        stats = manager.get_performance_stats()

        # Total: 12 frames in 0.37s = 32.4 FPS
        assert stats["avg_fps"] > 30.0
        assert stats["avg_fps"] < 35.0

        # Average latency: 0.37s / 12 frames = 30.8 ms/frame
        assert stats["avg_latency_ms"] > 25.0
        assert stats["avg_latency_ms"] < 35.0

    def test_performance_stats_rolling_window(self, temp_model_dir, mock_yolo_model):
        """Test that performance stats use rolling window (last 100)."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")
        manager.model = mock_yolo_model
        manager.model_info = ModelInfo(
            path=Path("test.pt"),
            format="pt",
            precision="fp32",
            input_size=(640, 640),
            vram_mb=1024,
            device="cpu",
        )

        # Add 150 measurements (exceeds 100 limit)
        for i in range(150):
            manager.inference_times.append(0.1)
            manager.batch_sizes.append(1)

        # Should only keep last 100
        assert len(manager.inference_times) == 100
        assert len(manager.batch_sizes) == 100


class TestModelRepresentation:
    """Test string representations."""

    def test_repr_without_model(self, temp_model_dir):
        """Test __repr__ without loaded model."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")
        repr_str = repr(manager)

        assert "ModelManager" in repr_str
        assert "cpu" in repr_str
        assert "None" in repr_str

    def test_repr_with_model(self, temp_model_dir, mock_yolo_model):
        """Test __repr__ with loaded model."""
        manager = ModelManager(model_dir=temp_model_dir, device="cuda:0", warmup=False)
        manager.model = mock_yolo_model
        manager.loaded_model_name = "yolo11n"
        manager.model_info = ModelInfo(
            path=Path("test.engine"),
            format="engine",
            precision="fp16",
            input_size=(640, 640),
            vram_mb=2048,
            device="cuda:0",
        )

        repr_str = repr(manager)

        assert "yolo11n" in repr_str
        assert "cuda:0" in repr_str
        assert "engine" in repr_str


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_gpu_id(self):
        """Test invalid GPU ID raises error."""
        with patch("core.model_manager.TORCH_AVAILABLE", True):
            with patch("core.model_manager.torch.cuda.is_available", return_value=True):
                with patch("core.model_manager.torch.cuda.device_count", return_value=1):
                    with pytest.raises(RuntimeError, match="GPU 5 not found"):
                        ModelManager(model_dir=".", device="cuda:5")

    def test_vram_estimation(self, temp_model_dir):
        """Test VRAM usage estimation."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")

        # Create mock model file
        model_path = temp_model_dir / "test.pt"
        model_path.write_bytes(b"0" * (100 * 1024 * 1024))  # 100 MB file

        vram_mb = manager._estimate_vram_usage(
            model_path=model_path, precision="fp32", input_size=(640, 640)
        )

        # Should estimate model + activations + overhead
        assert vram_mb > 100  # At least model size
        assert vram_mb < 1000  # Reasonable upper bound

    def test_vram_estimation_fp16_reduction(self, temp_model_dir):
        """Test FP16 reduces VRAM estimation."""
        manager = ModelManager(model_dir=temp_model_dir, device="cpu")

        model_path = temp_model_dir / "test.pt"
        model_path.write_bytes(b"0" * (100 * 1024 * 1024))

        vram_fp32 = manager._estimate_vram_usage(model_path, "fp32", (640, 640))
        vram_fp16 = manager._estimate_vram_usage(model_path, "fp16", (640, 640))

        # FP16 should use less VRAM
        assert vram_fp16 < vram_fp32


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
