"""
Unit tests for Config system - Hardware profiles and settings.

Test coverage:
- Hardware profile creation and validation
- Auto-detection of hardware capabilities
- Profile selection (dev_pi, prod_rtx3090, debug)
- Environment variable overrides
- Resolution-specific optimization
- Config save/load functionality

Author: ml-specialist agent
Date: 2025-10-24
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import (
    PROFILES,
    TORCH_AVAILABLE,
    Config,
    HardwareProfile,
    get_config,
    reset_config,
    set_config,
)


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset global config before each test."""
    reset_config()
    yield
    reset_config()


class TestHardwareProfile:
    """Test HardwareProfile dataclass."""

    def test_profile_creation(self):
        """Test creating hardware profile."""
        profile = HardwareProfile(
            name="test", device="cuda:0", use_tensorrt=True, use_fp16=True, max_batch_size=8
        )

        assert profile.name == "test"
        assert profile.device == "cuda:0"
        assert profile.use_tensorrt is True
        assert profile.max_batch_size == 8

    def test_profile_defaults(self):
        """Test profile default values."""
        profile = HardwareProfile(name="test", device="cpu")

        assert profile.use_tensorrt is True  # Default
        assert profile.enable_reid is False  # Default
        assert profile.log_level == "INFO"  # Default

    def test_profile_cuda_without_torch_fallback(self):
        """Test profile falls back to CPU if CUDA requested but unavailable."""
        with patch("core.config.TORCH_AVAILABLE", False):
            profile = HardwareProfile(name="test", device="cuda:0")
            # Should fall back to CPU in __post_init__
            assert profile.device == "cpu"
            assert profile.use_tensorrt is False

    def test_profile_to_dict(self):
        """Test converting profile to dictionary."""
        profile = HardwareProfile(name="test", device="cuda:0", max_batch_size=16)

        profile_dict = profile.to_dict()

        assert isinstance(profile_dict, dict)
        assert profile_dict["name"] == "test"
        assert profile_dict["device"] == "cuda:0"
        assert profile_dict["max_batch_size"] == 16


class TestPredefinedProfiles:
    """Test predefined hardware profiles."""

    def test_all_profiles_exist(self):
        """Test that all expected profiles are defined."""
        expected_profiles = ["dev_pi", "prod_rtx3090", "debug"]

        for profile_name in expected_profiles:
            assert profile_name in PROFILES

    def test_dev_pi_profile(self):
        """Test dev_pi profile settings."""
        profile = PROFILES["dev_pi"]

        assert profile.device == "cpu"
        assert profile.use_tensorrt is False
        assert profile.use_fp16 is False
        assert profile.max_batch_size == 1
        assert profile.enable_optical_flow is False
        assert profile.enable_reid is False

    def test_prod_rtx3090_profile(self):
        """Test prod_rtx3090 profile settings."""
        profile = PROFILES["prod_rtx3090"]

        assert profile.device == "cuda:0"
        assert profile.use_tensorrt is True
        assert profile.use_fp16 is True
        assert profile.max_batch_size == 8
        assert profile.enable_optical_flow is True
        assert profile.enable_reid is True
        assert profile.vram_limit_gb == 20.0

    def test_debug_profile(self):
        """Test debug profile settings."""
        profile = PROFILES["debug"]

        assert profile.use_tensorrt is False  # Disabled for debugging
        assert profile.max_batch_size == 1
        assert profile.log_level == "DEBUG"


class TestConfigInitialization:
    """Test Config initialization."""

    def test_init_from_profile(self):
        """Test creating config from hardware profile."""
        profile = PROFILES["dev_pi"]
        config = Config(profile)

        assert config.name == "dev_pi"
        assert config.device == "cpu"
        assert config.use_tensorrt is False

    def test_init_creates_directories(self, temp_config_dir):
        """Test that initialization creates required directories."""
        with patch.dict(
            os.environ,
            {
                "FUNGEN_MODEL_DIR": str(temp_config_dir / "models"),
                "FUNGEN_OUTPUT_DIR": str(temp_config_dir / "output"),
                "FUNGEN_CACHE_DIR": str(temp_config_dir / "cache"),
            },
        ):
            profile = PROFILES["dev_pi"]
            config = Config(profile)

            assert config.model_dir.exists()
            assert config.output_dir.exists()
            assert config.cache_dir.exists()

    def test_from_profile_success(self):
        """Test creating config from profile name."""
        config = Config.from_profile("dev_pi")

        assert config.name == "dev_pi"
        assert isinstance(config, Config)

    def test_from_profile_invalid_name(self):
        """Test creating config with invalid profile name raises error."""
        with pytest.raises(ValueError, match="Unknown profile"):
            Config.from_profile("nonexistent_profile")

    def test_environment_variable_overrides(self):
        """Test environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "FUNGEN_TRACKER": "bytetrack",
                "FUNGEN_CONF_THRESHOLD": "0.5",
                "FUNGEN_IOU_THRESHOLD": "0.7",
            },
        ):
            profile = PROFILES["dev_pi"]
            config = Config(profile)

            assert config.default_tracker == "bytetrack"
            assert config.conf_threshold == 0.5
            assert config.iou_threshold == 0.7


class TestConfigAutoDetection:
    """Test automatic hardware detection."""

    def test_auto_detect_uses_env_variable(self):
        """Test auto-detect uses FUNGEN_PROFILE env variable."""
        with patch.dict(os.environ, {"FUNGEN_PROFILE": "debug"}):
            config = Config.auto_detect()
            assert config.name == "debug"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires PyTorch")
    def test_auto_detect_cuda_available(self):
        """Test auto-detect selects prod profile with CUDA."""
        with patch("core.config.TORCH_AVAILABLE", True):
            with patch("core.config.torch.cuda.is_available", return_value=True):
                with patch("core.config.torch.cuda.get_device_name", return_value="RTX 3090"):
                    mock_props = Mock()
                    mock_props.total_memory = 24 * 1024**3  # 24 GB
                    with patch(
                        "core.config.torch.cuda.get_device_properties", return_value=mock_props
                    ):
                        config = Config.auto_detect()
                        assert config.name == "prod_rtx3090"

    def test_auto_detect_no_cuda_fallback(self):
        """Test auto-detect falls back to dev_pi without CUDA."""
        with patch("core.config.TORCH_AVAILABLE", False):
            config = Config.auto_detect()
            assert config.name == "dev_pi"
            assert config.device == "cpu"

    def test_auto_detect_insufficient_vram(self):
        """Test auto-detect uses debug profile for GPUs with <20GB VRAM."""
        with patch("core.config.TORCH_AVAILABLE", True):
            with patch("core.config.torch.cuda.is_available", return_value=True):
                with patch("core.config.torch.cuda.get_device_name", return_value="GTX 1080"):
                    mock_props = Mock()
                    mock_props.total_memory = 8 * 1024**3  # 8 GB
                    with patch(
                        "core.config.torch.cuda.get_device_properties", return_value=mock_props
                    ):
                        config = Config.auto_detect()
                        assert config.name == "debug"


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_success(self):
        """Test validation passes for valid config."""
        config = Config.from_profile("dev_pi")
        assert config.validate() is True

    def test_validate_cuda_without_torch_raises_error(self):
        """Test validation fails for CUDA device without PyTorch."""
        with patch("core.config.TORCH_AVAILABLE", False):
            profile = HardwareProfile(name="test", device="cuda:0")
            profile.device = "cuda:0"  # Override post_init fallback
            config = Config(profile)

            with pytest.raises(ValueError, match="CUDA device requested"):
                config.validate()

    def test_validate_invalid_batch_size(self):
        """Test validation fails for invalid batch size."""
        config = Config.from_profile("dev_pi")
        config.max_batch_size = 0

        with pytest.raises(ValueError, match="Invalid batch size"):
            config.validate()

    def test_validate_invalid_confidence_threshold(self):
        """Test validation fails for invalid confidence threshold."""
        config = Config.from_profile("dev_pi")
        config.conf_threshold = 1.5  # Invalid (>1.0)

        with pytest.raises(ValueError, match="Invalid confidence threshold"):
            config.validate()

    def test_validate_invalid_iou_threshold(self):
        """Test validation fails for invalid IoU threshold."""
        config = Config.from_profile("dev_pi")
        config.iou_threshold = -0.1  # Invalid (<0.0)

        with pytest.raises(ValueError, match="Invalid IoU threshold"):
            config.validate()


class TestConfigSaveLoad:
    """Test configuration save/load."""

    def test_save_config(self, temp_config_dir):
        """Test saving configuration to JSON."""
        config = Config.from_profile("dev_pi")
        save_path = temp_config_dir / "config.json"

        config.save(save_path)

        assert save_path.exists()

        # Verify JSON structure
        with open(save_path) as f:
            data = json.load(f)

        assert "profile" in data
        assert data["profile"]["name"] == "dev_pi"

    def test_load_config(self, temp_config_dir):
        """Test loading configuration from JSON."""
        # Create and save config
        original_config = Config.from_profile("prod_rtx3090")
        save_path = temp_config_dir / "config.json"
        original_config.save(save_path)

        # Load config
        loaded_config = Config.load(save_path)

        assert loaded_config.name == "prod_rtx3090"
        assert loaded_config.device == original_config.device
        assert loaded_config.max_batch_size == original_config.max_batch_size

    def test_save_load_preserves_custom_settings(self, temp_config_dir):
        """Test save/load preserves custom settings."""
        config = Config.from_profile("dev_pi")
        config.conf_threshold = 0.3
        config.default_tracker = "botsort"

        save_path = temp_config_dir / "config.json"
        config.save(save_path)

        loaded_config = Config.load(save_path)

        assert loaded_config.conf_threshold == 0.3
        assert loaded_config.default_tracker == "botsort"


class TestResolutionOptimization:
    """Test resolution-specific optimization."""

    def test_optimization_1080p(self):
        """Test optimal settings for 1080p."""
        config = Config.from_profile("prod_rtx3090")
        settings = config.get_optimal_settings_for_resolution(1920, 1080)

        assert settings["batch_size"] == 8  # Full batch
        assert settings["resize_factor"] == 1.0
        assert settings["target_fps"] == 100

    def test_optimization_4k(self):
        """Test optimal settings for 4K."""
        config = Config.from_profile("prod_rtx3090")
        settings = config.get_optimal_settings_for_resolution(3840, 2160)

        assert settings["batch_size"] == 4  # Half batch
        assert settings["resize_factor"] == 0.75
        assert settings["target_fps"] == 60

    def test_optimization_8k(self):
        """Test optimal settings for 8K."""
        config = Config.from_profile("prod_rtx3090")
        settings = config.get_optimal_settings_for_resolution(7680, 4320)

        assert settings["batch_size"] == 2  # Quarter batch
        assert settings["resize_factor"] == 0.5
        assert settings["target_fps"] == 30


class TestGlobalConfig:
    """Test global configuration singleton."""

    def test_get_config_creates_instance(self):
        """Test get_config creates global instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_get_config_returns_same_instance(self):
        """Test get_config returns same instance on multiple calls."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config_updates_global(self):
        """Test set_config updates global instance."""
        custom_config = Config.from_profile("debug")
        set_config(custom_config)

        global_config = get_config()
        assert global_config.name == "debug"

    def test_reset_config_clears_global(self):
        """Test reset_config clears global instance."""
        _ = get_config()  # Create instance
        reset_config()

        # Next call should create new instance
        new_config = get_config()
        assert isinstance(new_config, Config)


class TestConfigRepresentation:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ output."""
        config = Config.from_profile("prod_rtx3090")
        repr_str = repr(config)

        assert "Config" in repr_str
        assert "prod_rtx3090" in repr_str
        assert "cuda:0" in repr_str
        assert "tensorrt=True" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
