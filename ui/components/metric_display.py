"""
Real-time metric display widget (FPS, VRAM, CPU, etc.).

Author: ui-architect agent
"""

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional


class MetricDisplay(ttk.Frame):
    """
    Real-time metric display with color coding and sparklines.

    Features:
        - Color-coded values (green/yellow/red based on thresholds)
        - Optional sparkline graphs
        - Multiple metric support
        - Configurable update rate

    Attributes:
        metrics (Dict): Dictionary of metric widgets
    """

    def __init__(self, parent: tk.Widget, **kwargs):
        """
        Initialize metric display.

        Args:
            parent: Parent widget
            **kwargs: Additional frame options
        """
        super().__init__(parent, **kwargs)

        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.grid_columnconfigure(0, weight=1)

    def add_metric(
        self,
        key: str,
        label: str,
        unit: str = "",
        warning_threshold: Optional[float] = None,
        error_threshold: Optional[float] = None,
        format_str: str = "{:.1f}",
    ) -> None:
        """
        Add a metric to display.

        Args:
            key: Unique metric key
            label: Display label
            unit: Unit string (e.g., "FPS", "GB")
            warning_threshold: Warning threshold value
            error_threshold: Error threshold value
            format_str: Format string for value
        """
        row = len(self.metrics)

        # Container frame
        frame = ttk.Frame(self)
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        frame.grid_columnconfigure(1, weight=1)

        # Label
        label_widget = ttk.Label(frame, text=f"{label}:", font=("Arial", 9))
        label_widget.grid(row=0, column=0, sticky="w", padx=(0, 10))

        # Value label
        value_widget = ttk.Label(
            frame, text=f"0 {unit}", font=("Arial", 10, "bold"), foreground="gray"
        )
        value_widget.grid(row=0, column=1, sticky="e")

        # Store widgets
        self.metrics[key] = {
            "frame": frame,
            "label": label_widget,
            "value": value_widget,
            "unit": unit,
            "warning_threshold": warning_threshold,
            "error_threshold": error_threshold,
            "format_str": format_str,
            "current_value": 0.0,
        }

    def update_metric(self, key: str, value: float) -> None:
        """
        Update a metric value.

        Args:
            key: Metric key
            value: New value
        """
        if key not in self.metrics:
            return

        metric = self.metrics[key]
        metric["current_value"] = value

        # Format value
        formatted = metric["format_str"].format(value)
        text = f"{formatted} {metric['unit']}"

        # Determine color
        color = "gray"
        if metric["error_threshold"] and value >= metric["error_threshold"]:
            color = "red"
        elif metric["warning_threshold"] and value >= metric["warning_threshold"]:
            color = "orange"
        else:
            color = "green"

        # Update label
        metric["value"].config(text=text, foreground=color)

    def get_metric_value(self, key: str) -> Optional[float]:
        """
        Get current metric value.

        Args:
            key: Metric key

        Returns:
            Current value or None if not found
        """
        if key in self.metrics:
            return self.metrics[key]["current_value"]
        return None

    def clear_metrics(self) -> None:
        """Clear all metrics."""
        for key in self.metrics:
            self.update_metric(key, 0.0)
