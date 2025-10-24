"""
FunGen Rewrite - Enhanced UI Widgets

Comprehensive collection of polished UI components including:
- Enhanced progress bars with color coding and animations
- Tooltips for all UI elements
- Status indicators (GPU health, FPS counter, VRAM usage)
- Custom styled components
- Hardware monitoring widgets

Author: ui-enhancer agent
Date: 2025-10-24
Platform: Cross-platform (Pi + RTX 3090)
"""

import math
import time
import tkinter as tk
from dataclasses import dataclass
from enum import Enum
from tkinter import ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

# ============================================================================
# Color Schemes and Constants
# ============================================================================


class ColorScheme(Enum):
    """Pre-defined color schemes for widgets."""

    SUCCESS = ("#2ecc71", "#27ae60")  # Green
    WARNING = ("#f39c12", "#e67e22")  # Orange
    ERROR = ("#e74c3c", "#c0392b")  # Red
    INFO = ("#3498db", "#2980b9")  # Blue
    NEUTRAL = ("#95a5a6", "#7f8c8d")  # Gray


@dataclass
class ThresholdConfig:
    """Configuration for threshold-based color coding."""

    warning: Optional[float] = None
    critical: Optional[float] = None
    reverse: bool = False  # If True, lower is better (e.g., latency)


# ============================================================================
# Enhanced Progress Bar with Animations
# ============================================================================


class AnimatedProgressBar(ttk.Frame):
    """
    Enhanced progress bar with smooth animations and color coding.

    Features:
        - Smooth value transitions (easing)
        - Dynamic color coding based on progress/thresholds
        - Percentage display with customizable format
        - Speed indicator (FPS, MB/s, etc.)
        - Estimated time remaining
        - Indeterminate mode with animation
        - Gradient fill (if supported)

    Attributes:
        current_value (float): Current progress value
        target_value (float): Target progress value (for animation)
        max_value (float): Maximum value
        animation_speed (float): Animation speed (0-1, default 0.3)
    """

    def __init__(
        self,
        parent: tk.Widget,
        label: Optional[str] = None,
        max_value: float = 100,
        width: int = 300,
        height: int = 20,
        show_percentage: bool = True,
        show_speed: bool = False,
        show_eta: bool = False,
        color_mode: str = "auto",  # "auto", "static", "gradient"
        thresholds: Optional[ThresholdConfig] = None,
        animation_speed: float = 0.3,
        **kwargs,
    ):
        """
        Initialize animated progress bar.

        Args:
            parent: Parent widget
            label: Optional label text
            max_value: Maximum progress value
            width: Bar width in pixels
            height: Bar height in pixels
            show_percentage: Show percentage display
            show_speed: Show processing speed
            show_eta: Show estimated time remaining
            color_mode: Color coding mode
            thresholds: Threshold configuration for color coding
            animation_speed: Animation speed (0-1)
            **kwargs: Additional frame options
        """
        super().__init__(parent, **kwargs)

        self.max_value = max_value
        self.current_value = 0.0
        self.target_value = 0.0
        self.animation_speed = animation_speed
        self.show_percentage = show_percentage
        self.show_speed = show_speed
        self.show_eta = show_eta
        self.color_mode = color_mode
        self.thresholds = thresholds or ThresholdConfig()

        # Performance tracking
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.last_value = 0.0
        self.speed = 0.0

        # Animation state
        self.animation_id: Optional[str] = None
        self.is_animating = False

        # Configure grid
        self.grid_columnconfigure(1, weight=1)

        # Create widgets
        self._create_widgets(label, width, height)

    def _create_widgets(self, label: Optional[str], width: int, height: int) -> None:
        """Create all sub-widgets."""
        row = 0

        # Label
        if label:
            self.label = ttk.Label(self, text=label, font=("Arial", 9, "bold"))
            self.label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 3))
            row += 1
        else:
            self.label = None

        # Progress bar container (using Canvas for custom rendering)
        self.canvas = tk.Canvas(
            self, width=width, height=height, highlightthickness=0, bg="#2b2b2b"
        )
        self.canvas.grid(row=row, column=0, columnspan=2, sticky="ew", padx=(0, 5))

        # Draw background
        self.bg_rect = self.canvas.create_rectangle(
            0, 0, width, height, fill="#2b2b2b", outline="#555555", width=1
        )

        # Progress fill
        self.fill_rect = self.canvas.create_rectangle(0, 0, 0, height, fill="#3498db", outline="")

        # Percentage text (on canvas)
        if self.show_percentage:
            self.percent_text = self.canvas.create_text(
                width // 2, height // 2, text="0%", fill="white", font=("Arial", 9, "bold")
            )
        else:
            self.percent_text = None

        # Info labels column
        info_col = 2

        # Speed indicator
        if self.show_speed:
            self.speed_label = ttk.Label(self, text="0 it/s", font=("Arial", 8), foreground="gray")
            self.speed_label.grid(row=row, column=info_col, sticky="e", padx=(5, 0))
            row += 1
        else:
            self.speed_label = None

        # ETA display
        if self.show_eta:
            self.eta_label = ttk.Label(
                self, text="ETA: --:--", font=("Arial", 8), foreground="gray"
            )
            self.eta_label.grid(
                row=row if not self.show_speed else row - 1,
                column=info_col if self.show_speed else info_col,
                sticky="e" if self.show_speed else "ne",
                padx=(5, 0),
                pady=(3, 0) if self.show_speed else (0, 0),
            )
        else:
            self.eta_label = None

    def set_value(self, value: float, animate: bool = True) -> None:
        """
        Set progress value with optional animation.

        Args:
            value: New progress value
            animate: Whether to animate the transition
        """
        value = max(0, min(value, self.max_value))
        self.target_value = value

        if animate:
            self._start_animation()
        else:
            self.current_value = value
            self._update_display()

    def _start_animation(self) -> None:
        """Start smooth animation to target value."""
        if not self.is_animating:
            self.is_animating = True
            self._animate_step()

    def _animate_step(self) -> None:
        """Perform one animation step."""
        if not self.is_animating:
            return

        # Calculate difference
        diff = self.target_value - self.current_value

        # Apply easing (exponential ease-out)
        step = diff * self.animation_speed

        # Stop if very close
        if abs(diff) < 0.1:
            self.current_value = self.target_value
            self.is_animating = False
        else:
            self.current_value += step

        # Update display
        self._update_display()

        # Schedule next frame
        if self.is_animating:
            self.animation_id = self.after(16, self._animate_step)  # ~60 FPS

    def _update_display(self) -> None:
        """Update visual display with current value."""
        # Calculate percentage and width
        percent = (self.current_value / self.max_value) * 100
        width = self.canvas.winfo_width()
        if width <= 1:
            width = 300  # Default before widget is rendered
        fill_width = int((self.current_value / self.max_value) * width)

        # Update fill rectangle
        self.canvas.coords(self.fill_rect, 0, 0, fill_width, self.canvas.winfo_height() or 20)

        # Update color based on thresholds or progress
        fill_color = self._get_color(percent)
        self.canvas.itemconfig(self.fill_rect, fill=fill_color)

        # Update percentage text
        if self.percent_text:
            self.canvas.itemconfig(self.percent_text, text=f"{percent:.1f}%")

        # Update speed
        if self.show_speed:
            current_time = time.time()
            dt = current_time - self.last_update_time
            if dt > 0:
                dv = self.current_value - self.last_value
                self.speed = dv / dt
                self.speed_label.config(text=f"{self.speed:.1f} it/s")
            self.last_update_time = current_time
            self.last_value = self.current_value

        # Update ETA
        if self.show_eta and self.speed > 0:
            remaining = self.max_value - self.current_value
            eta_seconds = remaining / self.speed
            eta_minutes = int(eta_seconds // 60)
            eta_secs = int(eta_seconds % 60)
            self.eta_label.config(text=f"ETA: {eta_minutes:02d}:{eta_secs:02d}")

    def _get_color(self, percent: float) -> str:
        """
        Get fill color based on progress and thresholds.

        Args:
            percent: Current percentage

        Returns:
            Hex color string
        """
        if self.color_mode == "static":
            return "#3498db"  # Blue

        # Threshold-based coloring
        if self.thresholds.critical and percent >= self.thresholds.critical:
            return "#e74c3c" if not self.thresholds.reverse else "#2ecc71"
        elif self.thresholds.warning and percent >= self.thresholds.warning:
            return "#f39c12"

        # Auto color based on progress
        if percent < 33:
            return "#e74c3c"  # Red
        elif percent < 66:
            return "#f39c12"  # Orange
        else:
            return "#2ecc71"  # Green

    def set_indeterminate(self, active: bool = True) -> None:
        """
        Toggle indeterminate mode with pulsing animation.

        Args:
            active: Whether to activate indeterminate mode
        """
        if active:
            self.is_animating = False
            self._pulse_animation()
        else:
            if self.animation_id:
                self.after_cancel(self.animation_id)

    def _pulse_animation(self) -> None:
        """Pulsing animation for indeterminate mode."""
        # Create pulsing effect
        t = time.time() * 2
        pulse = (math.sin(t) + 1) / 2  # 0 to 1
        width = self.canvas.winfo_width() or 300
        fill_width = int(width * (0.3 + pulse * 0.4))

        self.canvas.coords(self.fill_rect, 0, 0, fill_width, self.canvas.winfo_height() or 20)
        self.canvas.itemconfig(self.fill_rect, fill="#3498db")

        if self.percent_text:
            self.canvas.itemconfig(self.percent_text, text="...")

        # Continue pulsing
        self.animation_id = self.after(50, self._pulse_animation)

    def reset(self) -> None:
        """Reset progress bar to zero."""
        self.current_value = 0.0
        self.target_value = 0.0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.last_value = 0.0
        self.speed = 0.0
        self._update_display()


# ============================================================================
# Hardware Status Indicators
# ============================================================================


class GPUHealthIndicator(ttk.Frame):
    """
    GPU health status indicator with real-time monitoring.

    Features:
        - Temperature monitoring with color coding
        - VRAM usage display
        - GPU utilization percentage
        - Power consumption
        - Visual health status (good/warning/critical)

    Attributes:
        temperature (float): GPU temperature in Celsius
        vram_used (float): VRAM used in GB
        vram_total (float): Total VRAM in GB
        utilization (float): GPU utilization (0-100%)
        power_draw (float): Power draw in watts
    """

    def __init__(
        self,
        parent: tk.Widget,
        show_temperature: bool = True,
        show_vram: bool = True,
        show_utilization: bool = True,
        show_power: bool = False,
        **kwargs,
    ):
        """
        Initialize GPU health indicator.

        Args:
            parent: Parent widget
            show_temperature: Show temperature reading
            show_vram: Show VRAM usage
            show_utilization: Show GPU utilization
            show_power: Show power consumption
            **kwargs: Additional frame options
        """
        super().__init__(parent, **kwargs)

        # State
        self.temperature = 0.0
        self.vram_used = 0.0
        self.vram_total = 24.0  # Default for RTX 3090
        self.utilization = 0.0
        self.power_draw = 0.0

        # Configure
        self.grid_columnconfigure(1, weight=1)

        # Create widgets
        row = 0

        # Title
        title_label = ttk.Label(self, text="GPU Health", font=("Arial", 10, "bold"))
        title_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1

        # Status indicator (colored circle)
        self.status_canvas = tk.Canvas(
            self, width=20, height=20, highlightthickness=0, bg=self.cget("background")
        )
        self.status_canvas.grid(row=row, column=0, rowspan=4, padx=(0, 10), sticky="n")
        self.status_indicator = self.status_canvas.create_oval(
            3, 3, 17, 17, fill="#2ecc71", outline="#27ae60", width=2
        )

        # Temperature
        if show_temperature:
            ttk.Label(self, text="Temp:", font=("Arial", 9)).grid(row=row, column=1, sticky="w")
            self.temp_label = ttk.Label(
                self, text="-- °C", font=("Arial", 9, "bold"), foreground="gray"
            )
            self.temp_label.grid(row=row, column=2, sticky="e", padx=(5, 0))
            row += 1
        else:
            self.temp_label = None

        # VRAM
        if show_vram:
            ttk.Label(self, text="VRAM:", font=("Arial", 9)).grid(row=row, column=1, sticky="w")
            self.vram_label = ttk.Label(
                self, text="0.0 / 24.0 GB", font=("Arial", 9, "bold"), foreground="gray"
            )
            self.vram_label.grid(row=row, column=2, sticky="e", padx=(5, 0))
            row += 1
        else:
            self.vram_label = None

        # Utilization
        if show_utilization:
            ttk.Label(self, text="Usage:", font=("Arial", 9)).grid(row=row, column=1, sticky="w")
            self.util_label = ttk.Label(
                self, text="0%", font=("Arial", 9, "bold"), foreground="gray"
            )
            self.util_label.grid(row=row, column=2, sticky="e", padx=(5, 0))
            row += 1
        else:
            self.util_label = None

        # Power
        if show_power:
            ttk.Label(self, text="Power:", font=("Arial", 9)).grid(row=row, column=1, sticky="w")
            self.power_label = ttk.Label(
                self, text="0 W", font=("Arial", 9, "bold"), foreground="gray"
            )
            self.power_label.grid(row=row, column=2, sticky="e", padx=(5, 0))
        else:
            self.power_label = None

    def update_status(
        self,
        temperature: Optional[float] = None,
        vram_used: Optional[float] = None,
        vram_total: Optional[float] = None,
        utilization: Optional[float] = None,
        power_draw: Optional[float] = None,
    ) -> None:
        """
        Update GPU status.

        Args:
            temperature: GPU temperature (Celsius)
            vram_used: VRAM used (GB)
            vram_total: Total VRAM (GB)
            utilization: GPU utilization (0-100%)
            power_draw: Power consumption (Watts)
        """
        # Update values
        if temperature is not None:
            self.temperature = temperature
        if vram_used is not None:
            self.vram_used = vram_used
        if vram_total is not None:
            self.vram_total = vram_total
        if utilization is not None:
            self.utilization = utilization
        if power_draw is not None:
            self.power_draw = power_draw

        # Update temperature display
        if self.temp_label and temperature is not None:
            color = self._get_temp_color(self.temperature)
            self.temp_label.config(text=f"{self.temperature:.1f} °C", foreground=color)

        # Update VRAM display
        if self.vram_label:
            vram_percent = (self.vram_used / self.vram_total) * 100
            color = self._get_vram_color(vram_percent)
            self.vram_label.config(
                text=f"{self.vram_used:.1f} / {self.vram_total:.1f} GB", foreground=color
            )

        # Update utilization
        if self.util_label:
            color = (
                "green" if self.utilization > 70 else "orange" if self.utilization > 30 else "gray"
            )
            self.util_label.config(text=f"{self.utilization:.0f}%", foreground=color)

        # Update power
        if self.power_label:
            self.power_label.config(text=f"{self.power_draw:.0f} W")

        # Update overall status indicator
        self._update_health_indicator()

    def _get_temp_color(self, temp: float) -> str:
        """Get color based on temperature."""
        if temp >= 85:
            return "red"
        elif temp >= 75:
            return "orange"
        else:
            return "green"

    def _get_vram_color(self, percent: float) -> str:
        """Get color based on VRAM usage."""
        if percent >= 90:
            return "red"
        elif percent >= 75:
            return "orange"
        else:
            return "green"

    def _update_health_indicator(self) -> None:
        """Update overall health status indicator."""
        # Determine overall health
        status = "good"

        if self.temperature >= 85 or (self.vram_used / self.vram_total) >= 0.95:
            status = "critical"
        elif self.temperature >= 75 or (self.vram_used / self.vram_total) >= 0.85:
            status = "warning"

        # Update indicator color
        colors = {
            "good": ("#2ecc71", "#27ae60"),
            "warning": ("#f39c12", "#e67e22"),
            "critical": ("#e74c3c", "#c0392b"),
        }
        fill, outline = colors[status]
        self.status_canvas.itemconfig(self.status_indicator, fill=fill, outline=outline)


# ============================================================================
# FPS Counter Widget
# ============================================================================


class FPSCounter(ttk.Frame):
    """
    Real-time FPS counter with statistics.

    Features:
        - Current FPS display
        - Average FPS over time window
        - Min/Max FPS tracking
        - Frame time display
        - Visual graph (sparkline)
        - Color coding based on target FPS

    Attributes:
        current_fps (float): Current FPS
        avg_fps (float): Average FPS
        min_fps (float): Minimum FPS
        max_fps (float): Maximum FPS
        target_fps (float): Target FPS for color coding
    """

    def __init__(
        self,
        parent: tk.Widget,
        target_fps: float = 100.0,
        show_stats: bool = True,
        show_graph: bool = True,
        history_size: int = 60,
        **kwargs,
    ):
        """
        Initialize FPS counter.

        Args:
            parent: Parent widget
            target_fps: Target FPS for color coding
            show_stats: Show min/max/avg statistics
            show_graph: Show FPS graph
            history_size: Number of samples for history
            **kwargs: Additional frame options
        """
        super().__init__(parent, **kwargs)

        self.target_fps = target_fps
        self.show_stats = show_stats
        self.show_graph = show_graph
        self.history_size = history_size

        # State
        self.current_fps = 0.0
        self.fps_history: List[float] = []
        self.min_fps = 0.0
        self.max_fps = 0.0
        self.avg_fps = 0.0

        # Create widgets
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create all sub-widgets."""
        # Main FPS display
        fps_frame = ttk.Frame(self)
        fps_frame.pack(fill="x", pady=(0, 5))

        ttk.Label(fps_frame, text="FPS:", font=("Arial", 9)).pack(side="left")

        self.fps_label = ttk.Label(
            fps_frame, text="0.0", font=("Arial", 16, "bold"), foreground="gray"
        )
        self.fps_label.pack(side="left", padx=(5, 0))

        # Frame time
        self.frametime_label = ttk.Label(
            fps_frame, text="(0.0 ms)", font=("Arial", 8), foreground="gray"
        )
        self.frametime_label.pack(side="left", padx=(5, 0))

        # Statistics
        if self.show_stats:
            stats_frame = ttk.Frame(self)
            stats_frame.pack(fill="x", pady=(0, 5))

            # Avg
            ttk.Label(stats_frame, text="Avg:", font=("Arial", 8)).pack(side="left")
            self.avg_label = ttk.Label(
                stats_frame, text="0.0", font=("Arial", 8, "bold"), foreground="gray"
            )
            self.avg_label.pack(side="left", padx=(2, 10))

            # Min
            ttk.Label(stats_frame, text="Min:", font=("Arial", 8)).pack(side="left")
            self.min_label = ttk.Label(
                stats_frame, text="0.0", font=("Arial", 8, "bold"), foreground="gray"
            )
            self.min_label.pack(side="left", padx=(2, 10))

            # Max
            ttk.Label(stats_frame, text="Max:", font=("Arial", 8)).pack(side="left")
            self.max_label = ttk.Label(
                stats_frame, text="0.0", font=("Arial", 8, "bold"), foreground="gray"
            )
            self.max_label.pack(side="left", padx=(2, 0))

        # Graph
        if self.show_graph:
            self.graph_canvas = tk.Canvas(
                self, width=200, height=40, highlightthickness=0, bg="#2b2b2b"
            )
            self.graph_canvas.pack(fill="x")

    def update_fps(self, fps: float) -> None:
        """
        Update FPS value.

        Args:
            fps: New FPS value
        """
        self.current_fps = fps

        # Add to history
        self.fps_history.append(fps)
        if len(self.fps_history) > self.history_size:
            self.fps_history.pop(0)

        # Calculate statistics
        if self.fps_history:
            self.avg_fps = sum(self.fps_history) / len(self.fps_history)
            self.min_fps = min(self.fps_history)
            self.max_fps = max(self.fps_history)

        # Update display
        self._update_display()

    def _update_display(self) -> None:
        """Update visual display."""
        # Get color based on target
        color = self._get_fps_color(self.current_fps)

        # Update main FPS
        self.fps_label.config(text=f"{self.current_fps:.1f}", foreground=color)

        # Update frame time
        if self.current_fps > 0:
            frame_time = 1000.0 / self.current_fps
            self.frametime_label.config(text=f"({frame_time:.1f} ms)")

        # Update stats
        if self.show_stats:
            self.avg_label.config(text=f"{self.avg_fps:.1f}")
            self.min_label.config(text=f"{self.min_fps:.1f}")
            self.max_label.config(text=f"{self.max_fps:.1f}")

        # Update graph
        if self.show_graph:
            self._draw_graph()

    def _get_fps_color(self, fps: float) -> str:
        """Get color based on FPS relative to target."""
        percent = (fps / self.target_fps) * 100

        if percent >= 90:
            return "green"
        elif percent >= 60:
            return "orange"
        else:
            return "red"

    def _draw_graph(self) -> None:
        """Draw FPS history graph."""
        self.graph_canvas.delete("all")

        if len(self.fps_history) < 2:
            return

        width = self.graph_canvas.winfo_width() or 200
        height = self.graph_canvas.winfo_height() or 40

        # Calculate scale
        max_val = max(self.fps_history) if self.fps_history else self.target_fps
        max_val = max(max_val, self.target_fps * 1.1)  # At least 110% of target

        # Draw target line
        target_y = height - (self.target_fps / max_val) * height
        self.graph_canvas.create_line(0, target_y, width, target_y, fill="#555555", dash=(2, 2))

        # Draw FPS line
        points = []
        for i, fps in enumerate(self.fps_history):
            x = (i / max(len(self.fps_history) - 1, 1)) * width
            y = height - (fps / max_val) * height
            points.extend([x, y])

        if len(points) >= 4:
            self.graph_canvas.create_line(*points, fill="#3498db", width=2, smooth=True)

    def reset(self) -> None:
        """Reset FPS counter."""
        self.current_fps = 0.0
        self.fps_history.clear()
        self.min_fps = 0.0
        self.max_fps = 0.0
        self.avg_fps = 0.0
        self._update_display()


# ============================================================================
# Enhanced Tooltip with Rich Content
# ============================================================================


class RichTooltip:
    """
    Enhanced tooltip with support for rich content.

    Features:
        - Multi-line text with formatting
        - Keyboard shortcut display
        - Icons and symbols
        - Delayed appearance
        - Smart positioning (avoid screen edges)
        - Theme-aware styling

    Attributes:
        widget (tk.Widget): Widget this tooltip is attached to
        text (str): Tooltip text
        shortcut (Optional[str]): Keyboard shortcut
    """

    def __init__(
        self,
        widget: tk.Widget,
        text: str,
        shortcut: Optional[str] = None,
        delay: int = 500,
        wraplength: int = 300,
        **kwargs,
    ):
        """
        Initialize rich tooltip.

        Args:
            widget: Widget to attach tooltip to
            text: Tooltip text (supports \\n for newlines)
            shortcut: Optional keyboard shortcut to display
            delay: Delay before showing (milliseconds)
            wraplength: Maximum text width
            **kwargs: Additional styling options
        """
        self.widget = widget
        self.text = text
        self.shortcut = shortcut
        self.delay = delay
        self.wraplength = wraplength
        self.kwargs = kwargs

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

        # Create frame
        frame = tk.Frame(self.tooltip_window, background="#2b2b2b", relief=tk.SOLID, borderwidth=1)
        frame.pack()

        # Main text
        text_label = tk.Label(
            frame,
            text=self.text,
            justify=tk.LEFT,
            background="#2b2b2b",
            foreground="#ffffff",
            wraplength=self.wraplength,
            font=("Arial", 9),
            padx=8,
            pady=6,
        )
        text_label.pack()

        # Shortcut (if provided)
        if self.shortcut:
            shortcut_frame = tk.Frame(frame, background="#1e1e1e")
            shortcut_frame.pack(fill="x", padx=4, pady=(0, 4))

            tk.Label(
                shortcut_frame,
                text=f"⌨ {self.shortcut}",
                background="#1e1e1e",
                foreground="#95a5a6",
                font=("Arial", 8, "italic"),
                padx=4,
                pady=2,
            ).pack()

        # Position window (ensure it's on screen)
        self.tooltip_window.update_idletasks()
        width = self.tooltip_window.winfo_width()
        height = self.tooltip_window.winfo_height()
        screen_width = self.tooltip_window.winfo_screenwidth()
        screen_height = self.tooltip_window.winfo_screenheight()

        # Adjust if too far right
        if x + width > screen_width:
            x = screen_width - width - 10

        # Adjust if too far down
        if y + height > screen_height:
            y = self.widget.winfo_rooty() - height - 5

        self.tooltip_window.wm_geometry(f"+{x}+{y}")

    def _hide_tooltip(self) -> None:
        """Hide the tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def update_text(self, text: str, shortcut: Optional[str] = None) -> None:
        """
        Update tooltip content.

        Args:
            text: New tooltip text
            shortcut: New keyboard shortcut
        """
        self.text = text
        if shortcut is not None:
            self.shortcut = shortcut


# ============================================================================
# Export all widgets
# ============================================================================

__all__ = [
    "AnimatedProgressBar",
    "GPUHealthIndicator",
    "FPSCounter",
    "RichTooltip",
    "ColorScheme",
    "ThresholdConfig",
]
