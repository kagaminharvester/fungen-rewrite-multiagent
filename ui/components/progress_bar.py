"""
Enhanced progress bar widget with percentage display and color coding.

Author: ui-architect agent
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional


class ProgressBar(ttk.Frame):
    """
    Enhanced progress bar with percentage label and color coding.

    Features:
        - Automatic percentage display
        - Color-coded based on progress (red -> yellow -> green)
        - Optional label text
        - Supports indeterminate mode

    Attributes:
        progress (float): Current progress value (0-100)
        max_value (float): Maximum progress value
        color_mode (bool): Whether to use color coding
    """

    def __init__(
        self,
        parent: tk.Widget,
        label: Optional[str] = None,
        max_value: float = 100,
        color_mode: bool = True,
        **kwargs,
    ):
        """
        Initialize progress bar.

        Args:
            parent: Parent widget
            label: Optional label text
            max_value: Maximum progress value
            color_mode: Enable color coding
            **kwargs: Additional frame options
        """
        super().__init__(parent, **kwargs)

        self.max_value = max_value
        self.color_mode = color_mode
        self.progress = 0.0

        self.grid_columnconfigure(0, weight=1)

        # Label
        if label:
            self.label = ttk.Label(self, text=label, font=("Arial", 9))
            self.label.grid(row=0, column=0, sticky="w", pady=(0, 3))
            bar_row = 1
        else:
            self.label = None
            bar_row = 0

        # Progress bar
        self.progress_bar = ttk.Progressbar(self, mode="determinate", maximum=max_value)
        self.progress_bar.grid(row=bar_row, column=0, sticky="ew")

        # Percentage label
        self.percent_label = ttk.Label(self, text="0%", font=("Arial", 8), foreground="gray")
        self.percent_label.grid(row=bar_row, column=1, padx=(5, 0))

    def set_value(self, value: float) -> None:
        """
        Set progress value.

        Args:
            value: Progress value (0 to max_value)
        """
        self.progress = min(value, self.max_value)
        self.progress_bar["value"] = self.progress

        # Update percentage
        percent = (self.progress / self.max_value) * 100
        self.percent_label.config(text=f"{percent:.0f}%")

        # Update color if enabled
        if self.color_mode:
            if percent < 33:
                color = "red"
            elif percent < 66:
                color = "orange"
            else:
                color = "green"
            self.percent_label.config(foreground=color)

    def set_label(self, text: str) -> None:
        """Update label text."""
        if self.label:
            self.label.config(text=text)

    def set_indeterminate(self, active: bool = True) -> None:
        """
        Toggle indeterminate mode.

        Args:
            active: Whether to activate indeterminate mode
        """
        if active:
            self.progress_bar.config(mode="indeterminate")
            self.progress_bar.start()
            self.percent_label.config(text="...")
        else:
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate")
