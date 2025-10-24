"""
Unit tests for UI components.

Author: ui-architect agent
Date: 2025-10-24
"""

import json
import os
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.components.metric_display import MetricDisplay
from ui.components.progress_bar import ProgressBar
from ui.components.status_bar import StatusBar, StatusType
from ui.components.tooltip import Tooltip
from ui.event_handlers import EventHandler, ProgressTracker, format_percentage, format_time


class TestProgressBar:
    """Test ProgressBar component."""

    def setup_method(self):
        """Create root window for each test."""
        self.root = tk.Tk()
        self.root.withdraw()

    def teardown_method(self):
        """Destroy root window after each test."""
        self.root.destroy()

    def test_creation(self):
        """Test ProgressBar creation."""
        progress = ProgressBar(self.root, label="Test", max_value=100)
        assert progress.max_value == 100
        assert progress.progress == 0.0

    def test_set_value(self):
        """Test setting progress value."""
        progress = ProgressBar(self.root, max_value=100)
        progress.set_value(50)
        assert progress.progress == 50.0

    def test_max_value_clamp(self):
        """Test value clamping at max."""
        progress = ProgressBar(self.root, max_value=100)
        progress.set_value(150)
        assert progress.progress == 100.0

    def test_color_mode(self):
        """Test color mode functionality."""
        progress = ProgressBar(self.root, max_value=100, color_mode=True)
        progress.set_value(20)  # Should be red
        progress.set_value(50)  # Should be orange
        progress.set_value(80)  # Should be green
        assert progress.progress == 80.0

    def test_indeterminate_mode(self):
        """Test indeterminate mode."""
        progress = ProgressBar(self.root, max_value=100)
        progress.set_indeterminate(True)
        assert progress.progress_bar.cget("mode") == "indeterminate"
        progress.set_indeterminate(False)
        assert progress.progress_bar.cget("mode") == "determinate"


class TestTooltip:
    """Test Tooltip component."""

    def setup_method(self):
        """Create root window for each test."""
        self.root = tk.Tk()
        self.root.withdraw()

    def teardown_method(self):
        """Destroy root window after each test."""
        self.root.destroy()

    def test_creation(self):
        """Test Tooltip creation."""
        button = ttk.Button(self.root, text="Test")
        tooltip = Tooltip(button, "Test tooltip")
        assert tooltip.text == "Test tooltip"
        assert tooltip.widget == button

    def test_update_text(self):
        """Test updating tooltip text."""
        button = ttk.Button(self.root, text="Test")
        tooltip = Tooltip(button, "Original")
        tooltip.update_text("Updated")
        assert tooltip.text == "Updated"

    def test_tooltip_window_none_on_init(self):
        """Test tooltip window is None on initialization."""
        button = ttk.Button(self.root, text="Test")
        tooltip = Tooltip(button, "Test")
        assert tooltip.tooltip_window is None


class TestStatusBar:
    """Test StatusBar component."""

    def setup_method(self):
        """Create root window for each test."""
        self.root = tk.Tk()
        self.root.withdraw()

    def teardown_method(self):
        """Destroy root window after each test."""
        self.root.destroy()

    def test_creation(self):
        """Test StatusBar creation."""
        status = StatusBar(self.root)
        assert status.status_type == StatusType.INFO
        assert status.message == ""

    def test_set_status(self):
        """Test setting status message."""
        status = StatusBar(self.root)
        status.set_status("Test message", StatusType.SUCCESS)
        assert status.message == "Test message"
        assert status.status_type == StatusType.SUCCESS

    def test_clear(self):
        """Test clearing status."""
        status = StatusBar(self.root)
        status.set_status("Test", StatusType.ERROR)
        status.clear()
        assert status.message == "Ready"
        assert status.status_type == StatusType.INFO

    def test_add_info_field(self):
        """Test adding info fields."""
        status = StatusBar(self.root)
        status.add_info_field("fps", "FPS: 60")
        assert "fps" in status.info_labels

    def test_update_info_field(self):
        """Test updating info field."""
        status = StatusBar(self.root)
        status.add_info_field("fps", "FPS: 60")
        status.update_info_field("fps", "FPS: 75")
        assert status.info_labels["fps"].cget("text") == "FPS: 75"


class TestMetricDisplay:
    """Test MetricDisplay component."""

    def setup_method(self):
        """Create root window for each test."""
        self.root = tk.Tk()
        self.root.withdraw()

    def teardown_method(self):
        """Destroy root window after each test."""
        self.root.destroy()

    def test_creation(self):
        """Test MetricDisplay creation."""
        metrics = MetricDisplay(self.root)
        assert len(metrics.metrics) == 0

    def test_add_metric(self):
        """Test adding a metric."""
        metrics = MetricDisplay(self.root)
        metrics.add_metric("fps", "FPS", unit="", warning_threshold=50)
        assert "fps" in metrics.metrics

    def test_update_metric(self):
        """Test updating metric value."""
        metrics = MetricDisplay(self.root)
        metrics.add_metric("fps", "FPS", unit="")
        metrics.update_metric("fps", 75.5)
        assert metrics.get_metric_value("fps") == 75.5

    def test_clear_metrics(self):
        """Test clearing all metrics."""
        metrics = MetricDisplay(self.root)
        metrics.add_metric("fps", "FPS")
        metrics.update_metric("fps", 75.5)
        metrics.clear_metrics()
        assert metrics.get_metric_value("fps") == 0.0


class TestEventHandler:
    """Test EventHandler class."""

    def test_creation(self):
        """Test EventHandler creation."""
        handler = EventHandler()
        assert len(handler.callbacks) == 0
        assert handler.processing_thread is None

    def test_register_callback(self):
        """Test registering callbacks."""
        handler = EventHandler()

        def callback(data):
            pass

        handler.register_callback("progress", callback)
        assert "progress" in handler.callbacks
        assert callback in handler.callbacks["progress"]

    def test_unregister_callback(self):
        """Test unregistering callbacks."""
        handler = EventHandler()

        def callback(data):
            pass

        handler.register_callback("progress", callback)
        handler.unregister_callback("progress", callback)
        assert callback not in handler.callbacks["progress"]

    def test_emit_event(self):
        """Test event emission."""
        handler = EventHandler()
        called = {"value": False}

        def callback(data):
            called["value"] = True

        handler.register_callback("test", callback)
        handler.emit_event("test", {})
        assert called["value"] is True


class TestProgressTracker:
    """Test ProgressTracker class."""

    def test_creation(self):
        """Test ProgressTracker creation."""
        tracker = ProgressTracker()
        assert tracker.start_time is None
        assert tracker.frames_processed == 0

    def test_start(self):
        """Test starting tracker."""
        tracker = ProgressTracker()
        tracker.start(1000)
        assert tracker.start_time is not None
        assert tracker.total_frames == 1000

    def test_update(self):
        """Test updating progress."""
        tracker = ProgressTracker()
        tracker.start(1000)
        tracker.update(500, 30.0)
        assert tracker.frames_processed == 500
        assert len(tracker.fps_history) == 1

    def test_get_stats(self):
        """Test getting statistics."""
        tracker = ProgressTracker()
        tracker.start(1000)
        tracker.update(500, 30.0)
        stats = tracker.get_stats()
        assert "progress" in stats
        assert "avg_fps" in stats
        assert stats["progress"] == 50.0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_format_time(self):
        """Test time formatting."""
        assert format_time(0) == "00:00:00"
        assert format_time(61) == "00:01:01"
        assert format_time(3661) == "01:01:01"

    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(50.0) == "50.0%"
        assert format_percentage(33.333) == "33.3%"
        assert format_percentage(100.0) == "100.0%"


class TestAgentDashboardIntegration:
    """Integration tests for Agent Dashboard."""

    def setup_method(self):
        """Create root window and temp directory."""
        self.root = tk.Tk()
        self.root.withdraw()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup."""
        self.root.destroy()
        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_agent_dashboard_with_progress_files(self):
        """Test AgentDashboard with mock progress files."""
        from ui.agent_dashboard import AgentDashboard

        # Create mock progress file
        progress_file = Path(self.temp_dir) / "test-agent.json"
        with open(progress_file, "w") as f:
            json.dump(
                {
                    "agent": "test-agent",
                    "progress": 75,
                    "status": "in_progress",
                    "current_task": "Testing",
                    "timestamp": "2025-10-24T20:00:00Z",
                },
                f,
            )

        # Create dashboard
        dashboard = AgentDashboard(self.root, progress_dir=self.temp_dir)
        assert dashboard is not None


def test_all_imports():
    """Test that all UI modules can be imported."""
    from ui.agent_dashboard import AgentDashboard
    from ui.event_handlers import EventHandler
    from ui.main_window import MainWindow
    from ui.settings_panel import SettingsPanel

    assert MainWindow is not None
    assert AgentDashboard is not None
    assert SettingsPanel is not None
    assert EventHandler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
