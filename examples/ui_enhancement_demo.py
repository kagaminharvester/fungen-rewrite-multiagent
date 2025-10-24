"""
FunGen Rewrite - UI Enhancement Demo

Demonstrates the enhanced UI components including:
- Animated progress bars
- GPU health indicators
- FPS counters
- Rich tooltips
- Theme switching
- Smooth animations

Author: ui-enhancer agent
Date: 2025-10-24
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.animations import AnimationSequence, EasingType, FadeAnimation, SlideAnimation
from ui.themes import ThemeManager, ThemeType, get_theme_manager
from ui.widgets import (
    AnimatedProgressBar,
    FPSCounter,
    GPUHealthIndicator,
    RichTooltip,
    ThresholdConfig,
)


class UIEnhancementDemo(tk.Tk):
    """
    Demonstration of all enhanced UI components.

    Features:
        - Multiple animated progress bars
        - GPU health monitoring
        - FPS counter with graph
        - Theme switcher
        - Smooth animations
        - Rich tooltips on all elements
    """

    def __init__(self):
        """Initialize demo window."""
        super().__init__()

        self.title("FunGen UI Enhancement Demo")
        self.geometry("900x700")

        # Initialize theme manager
        self.theme_manager = get_theme_manager()
        self.theme_manager.apply_theme(self.theme_manager.current_theme, self)

        # Create UI
        self._create_ui()

        # Start demo animations
        self._start_demo()

    def _create_ui(self) -> None:
        """Create demo UI."""
        # Main container
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)

        # Title
        title = ttk.Label(main_frame, text="FunGen UI Enhancement Demo", font=("Arial", 18, "bold"))
        title.pack(pady=(0, 20))
        RichTooltip(title, "Demonstration of enhanced UI components", shortcut=None)

        # Theme switcher
        theme_frame = ttk.LabelFrame(main_frame, text="Theme Selection", padding=10)
        theme_frame.pack(fill="x", pady=(0, 15))

        themes = ["Dark", "Light", "High Contrast", "Nord", "Dracula"]
        for theme_name in themes:
            btn = ttk.Button(
                theme_frame, text=theme_name, command=lambda t=theme_name: self._switch_theme(t)
            )
            btn.pack(side="left", padx=5)
            RichTooltip(
                btn,
                f"Switch to {theme_name} theme",
                shortcut="Ctrl+T" if theme_name == "Dark" else None,
            )

        # Progress bars section
        progress_frame = ttk.LabelFrame(main_frame, text="Animated Progress Bars", padding=10)
        progress_frame.pack(fill="x", pady=(0, 15))

        # Standard progress bar with animation
        self.progress1 = AnimatedProgressBar(
            progress_frame,
            label="Video Processing",
            show_percentage=True,
            show_speed=True,
            show_eta=True,
            color_mode="auto",
        )
        self.progress1.pack(fill="x", pady=5)
        RichTooltip(
            self.progress1,
            "Animated progress bar with speed and ETA display\nSmooth transitions with easing",
            shortcut="Space (toggle)",
        )

        # Threshold-based progress bar
        self.progress2 = AnimatedProgressBar(
            progress_frame,
            label="VRAM Usage",
            show_percentage=True,
            thresholds=ThresholdConfig(warning=75.0, critical=90.0),
        )
        self.progress2.pack(fill="x", pady=5)
        RichTooltip(self.progress2, "Threshold-based color coding\nWarning: 75% | Critical: 90%")

        # Indeterminate progress bar
        self.progress3 = AnimatedProgressBar(
            progress_frame, label="Model Loading", show_percentage=True
        )
        self.progress3.pack(fill="x", pady=5)
        RichTooltip(self.progress3, "Indeterminate mode with pulsing animation")

        # Hardware monitoring section
        hardware_frame = ttk.LabelFrame(main_frame, text="Hardware Monitoring", padding=10)
        hardware_frame.pack(fill="x", pady=(0, 15))

        hw_container = ttk.Frame(hardware_frame)
        hw_container.pack(fill="x")

        # GPU health indicator
        self.gpu_health = GPUHealthIndicator(
            hw_container,
            show_temperature=True,
            show_vram=True,
            show_utilization=True,
            show_power=True,
        )
        self.gpu_health.pack(side="left", fill="y", padx=(0, 20))
        RichTooltip(
            self.gpu_health,
            "Real-time GPU health monitoring\n"
            "• Temperature with color coding\n"
            "• VRAM usage tracking\n"
            "• GPU utilization percentage\n"
            "• Power consumption",
            shortcut="Ctrl+G",
        )

        # FPS counter
        self.fps_counter = FPSCounter(
            hw_container, target_fps=100.0, show_stats=True, show_graph=True
        )
        self.fps_counter.pack(side="left", fill="both", expand=True)
        RichTooltip(
            self.fps_counter,
            "Real-time FPS counter\n"
            "• Current FPS with color coding\n"
            "• Min/Max/Average statistics\n"
            "• Frame time display\n"
            "• Historical graph (sparkline)",
            shortcut="Ctrl+F",
        )

        # Animation controls
        animation_frame = ttk.LabelFrame(main_frame, text="Animation Controls", padding=10)
        animation_frame.pack(fill="x", pady=(0, 15))

        # Control buttons
        btn_frame = ttk.Frame(animation_frame)
        btn_frame.pack(fill="x")

        start_btn = ttk.Button(btn_frame, text="Start Demo", command=self._start_demo)
        start_btn.pack(side="left", padx=5)
        RichTooltip(start_btn, "Start animated demo sequence", shortcut="Space")

        stop_btn = ttk.Button(btn_frame, text="Stop Demo", command=self._stop_demo)
        stop_btn.pack(side="left", padx=5)
        RichTooltip(stop_btn, "Stop all animations", shortcut="Ctrl+.")

        fade_btn = ttk.Button(btn_frame, text="Fade Animation", command=self._demo_fade)
        fade_btn.pack(side="left", padx=5)
        RichTooltip(fade_btn, "Demonstrate fade in/out animation")

        slide_btn = ttk.Button(btn_frame, text="Slide Animation", command=self._demo_slide)
        slide_btn.pack(side="left", padx=5)
        RichTooltip(slide_btn, "Demonstrate slide animation")

        # Animation target
        self.anim_target = ttk.Label(
            animation_frame,
            text="Animation Target Widget",
            background="#3498db",
            foreground="white",
            font=("Arial", 12, "bold"),
            padding=20,
        )
        self.anim_target.pack(pady=10)

        # Status bar
        status_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=1)
        status_frame.pack(fill="x", pady=(15, 0))

        self.status_label = ttk.Label(
            status_frame, text="Ready - Press 'Start Demo' to begin", padding=5
        )
        self.status_label.pack(fill="x")

        # Bind keyboard shortcuts
        self.bind("<space>", lambda e: self._start_demo())
        self.bind("<Control-t>", lambda e: self._cycle_theme())
        self.bind("<Control-period>", lambda e: self._stop_demo())

    def _switch_theme(self, theme_name: str) -> None:
        """Switch to specified theme."""
        try:
            self.theme_manager.switch_theme(theme_name, self)
            self.status_label.config(text=f"Theme changed to: {theme_name}")
        except ValueError as e:
            self.status_label.config(text=f"Error: {e}")

    def _cycle_theme(self) -> None:
        """Cycle through available themes."""
        self.theme_manager.toggle_dark_light(self)
        self.status_label.config(text=f"Theme: {self.theme_manager.current_theme.name}")

    def _start_demo(self) -> None:
        """Start animated demo."""
        self.status_label.config(text="Demo running...")

        # Animate progress bars
        self.progress1.set_value(0, animate=False)
        self.progress2.set_value(0, animate=False)
        self.progress3.set_indeterminate(True)

        self._animate_progress()
        self._animate_hardware()

    def _stop_demo(self) -> None:
        """Stop demo animations."""
        self.status_label.config(text="Demo stopped")
        self.progress3.set_indeterminate(False)

    def _animate_progress(self) -> None:
        """Animate progress bars."""
        import random

        # Animate first progress bar
        target = random.uniform(60, 100)
        self.progress1.set_value(target, animate=True)

        # Animate second progress bar
        target2 = random.uniform(50, 95)
        self.progress2.set_value(target2, animate=True)

        # Schedule next update
        self.after(2000, self._animate_progress)

    def _animate_hardware(self) -> None:
        """Animate hardware metrics."""
        import random

        # Update GPU health
        temp = random.uniform(60, 85)
        vram = random.uniform(8, 20)
        util = random.uniform(50, 100)
        power = random.uniform(200, 350)

        self.gpu_health.update_status(
            temperature=temp, vram_used=vram, vram_total=24.0, utilization=util, power_draw=power
        )

        # Update FPS
        fps = random.uniform(80, 120)
        self.fps_counter.update_fps(fps)

        # Schedule next update
        self.after(100, self._animate_hardware)

    def _demo_fade(self) -> None:
        """Demonstrate fade animation."""
        fade = FadeAnimation(self.anim_target, duration=0.5, easing=EasingType.EASE_IN_OUT)

        def fade_in():
            self.status_label.config(text="Fading in...")
            fade.fade_in(on_complete=lambda: self.status_label.config(text="Fade in complete"))

        def fade_out():
            self.status_label.config(text="Fading out...")
            fade.fade_out(on_complete=fade_in)

        fade_out()

    def _demo_slide(self) -> None:
        """Demonstrate slide animation."""
        slide = SlideAnimation(self.anim_target, duration=0.5, easing=EasingType.EASE_OUT)

        def slide_in():
            self.status_label.config(text="Sliding in from left...")
            slide.slide_in(
                "left", 200, on_complete=lambda: self.status_label.config(text="Slide complete")
            )

        def slide_out():
            self.status_label.config(text="Sliding out to right...")
            slide.slide_out("right", 200, on_complete=slide_in)

        slide_out()


def main():
    """Run the demo."""
    app = UIEnhancementDemo()
    app.mainloop()


if __name__ == "__main__":
    main()
