"""
Status bar widget with icon support.

Author: ui-architect agent
"""

import tkinter as tk
from enum import Enum
from tkinter import ttk
from typing import Optional


class StatusType(Enum):
    """Status message types."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class StatusBar(ttk.Frame):
    """
    Enhanced status bar with icon and color coding.

    Features:
        - Color-coded status messages
        - Icon indicators
        - Auto-clear timer
        - Multiple status fields

    Attributes:
        status_type (StatusType): Current status type
        message (str): Current status message
    """

    # Status colors
    STATUS_COLORS = {
        StatusType.INFO: "gray",
        StatusType.SUCCESS: "green",
        StatusType.WARNING: "orange",
        StatusType.ERROR: "red",
    }

    # Status icons (Unicode)
    STATUS_ICONS = {
        StatusType.INFO: "ℹ",
        StatusType.SUCCESS: "✓",
        StatusType.WARNING: "⚠",
        StatusType.ERROR: "✗",
    }

    def __init__(self, parent: tk.Widget, **kwargs):
        """
        Initialize status bar.

        Args:
            parent: Parent widget
            **kwargs: Additional frame options
        """
        super().__init__(parent, relief="sunken", borderwidth=1, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.status_type = StatusType.INFO
        self.message = ""
        self.clear_timer: Optional[str] = None

        # Icon label
        self.icon_label = ttk.Label(
            self, text=self.STATUS_ICONS[StatusType.INFO], font=("Arial", 10)
        )
        self.icon_label.pack(side="left", padx=(10, 5))

        # Message label
        self.message_label = ttk.Label(
            self, text="Ready", foreground=self.STATUS_COLORS[StatusType.INFO]
        )
        self.message_label.pack(side="left", fill="x", expand=True)

        # Additional info labels (right side)
        self.info_labels = {}

    def set_status(
        self,
        message: str,
        status_type: StatusType = StatusType.INFO,
        auto_clear: Optional[int] = None,
    ) -> None:
        """
        Set status message.

        Args:
            message: Status message
            status_type: Type of status
            auto_clear: Auto-clear after N milliseconds (None = no auto-clear)
        """
        self.message = message
        self.status_type = status_type

        # Update icon and color
        self.icon_label.config(text=self.STATUS_ICONS[status_type])
        self.message_label.config(text=message, foreground=self.STATUS_COLORS[status_type])

        # Cancel previous timer
        if self.clear_timer:
            self.after_cancel(self.clear_timer)
            self.clear_timer = None

        # Schedule auto-clear
        if auto_clear:
            self.clear_timer = self.after(auto_clear, self.clear)

    def clear(self) -> None:
        """Clear status message."""
        self.set_status("Ready", StatusType.INFO)

    def add_info_field(self, key: str, text: str = "") -> None:
        """
        Add an info field to the right side.

        Args:
            key: Unique key for the field
            text: Initial text
        """
        if key not in self.info_labels:
            label = ttk.Label(self, text=text, foreground="gray")
            label.pack(side="right", padx=10)
            self.info_labels[key] = label

    def update_info_field(self, key: str, text: str) -> None:
        """
        Update an info field.

        Args:
            key: Field key
            text: New text
        """
        if key in self.info_labels:
            self.info_labels[key].config(text=text)
