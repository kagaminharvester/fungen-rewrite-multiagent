"""
Tooltip widget for hover help text.

Author: ui-architect agent
"""

import tkinter as tk
from typing import Optional


class Tooltip:
    """
    Hover tooltip for widgets.

    Features:
        - Appears on mouse hover
        - Customizable delay
        - Auto-positioning
        - Wraps long text

    Usage:
        button = ttk.Button(root, text="Click me")
        Tooltip(button, "This button does something cool")
    """

    def __init__(self, widget: tk.Widget, text: str, delay: int = 500, wraplength: int = 200):
        """
        Initialize tooltip.

        Args:
            widget: Widget to attach tooltip to
            text: Tooltip text
            delay: Delay before showing (milliseconds)
            wraplength: Maximum text width before wrapping
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength

        self.tooltip_window: Optional[tk.Toplevel] = None
        self.schedule_id: Optional[str] = None

        # Bind events
        self.widget.bind("<Enter>", self._on_enter)
        self.widget.bind("<Leave>", self._on_leave)
        self.widget.bind("<Button>", self._on_leave)

    def _on_enter(self, event: tk.Event) -> None:
        """Handle mouse enter event."""
        self._cancel_schedule()
        self.schedule_id = self.widget.after(self.delay, self._show_tooltip)

    def _on_leave(self, event: tk.Event) -> None:
        """Handle mouse leave event."""
        self._cancel_schedule()
        self._hide_tooltip()

    def _cancel_schedule(self) -> None:
        """Cancel scheduled tooltip display."""
        if self.schedule_id:
            self.widget.after_cancel(self.schedule_id)
            self.schedule_id = None

    def _show_tooltip(self) -> None:
        """Display the tooltip."""
        if self.tooltip_window:
            return

        # Get widget position
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        # Create tooltip window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Create label
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            wraplength=self.wraplength,
            font=("Arial", 9),
            padx=5,
            pady=3,
        )
        label.pack()

    def _hide_tooltip(self) -> None:
        """Hide the tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def update_text(self, text: str) -> None:
        """
        Update tooltip text.

        Args:
            text: New tooltip text
        """
        self.text = text
