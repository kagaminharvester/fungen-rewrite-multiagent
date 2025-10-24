"""
FunGen Rewrite - UI Animations System

Smooth UI transitions and animations for enhanced user experience:
- Easing functions (ease-in, ease-out, elastic, bounce, etc.)
- Widget animations (fade, slide, scale, rotate)
- Value interpolation for smooth transitions
- Animation choreography and sequencing
- FPS-aware animations (60 FPS target)

Author: ui-enhancer agent
Date: 2025-10-24
Platform: Cross-platform (Pi + RTX 3090)
"""

import math
import time
import tkinter as tk
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

# ============================================================================
# Easing Functions
# ============================================================================


class EasingType(Enum):
    """Available easing functions."""

    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    EASE_IN_QUAD = "ease_in_quad"
    EASE_OUT_QUAD = "ease_out_quad"
    EASE_IN_OUT_QUAD = "ease_in_out_quad"
    EASE_IN_CUBIC = "ease_in_cubic"
    EASE_OUT_CUBIC = "ease_out_cubic"
    EASE_IN_OUT_CUBIC = "ease_in_out_cubic"
    ELASTIC = "elastic"
    BOUNCE = "bounce"


def linear(t: float) -> float:
    """Linear easing (no acceleration)."""
    return t


def ease_in(t: float) -> float:
    """Ease in (start slow, accelerate)."""
    return t * t


def ease_out(t: float) -> float:
    """Ease out (start fast, decelerate)."""
    return t * (2 - t)


def ease_in_out(t: float) -> float:
    """Ease in-out (slow start and end)."""
    return t * t * (3 - 2 * t)


def ease_in_quad(t: float) -> float:
    """Quadratic ease in."""
    return t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease out."""
    return t * (2 - t)


def ease_in_out_quad(t: float) -> float:
    """Quadratic ease in-out."""
    if t < 0.5:
        return 2 * t * t
    else:
        return -1 + (4 - 2 * t) * t


def ease_in_cubic(t: float) -> float:
    """Cubic ease in."""
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Cubic ease out."""
    t -= 1
    return t * t * t + 1


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease in-out."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        t -= 1
        return 4 * t * t * t + 1


def elastic(t: float) -> float:
    """Elastic easing (bouncy overshoot)."""
    if t == 0 or t == 1:
        return t
    p = 0.3
    s = p / 4
    return math.pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1


def bounce(t: float) -> float:
    """Bounce easing."""
    if t < 1 / 2.75:
        return 7.5625 * t * t
    elif t < 2 / 2.75:
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < 2.5 / 2.75:
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375


# Easing function registry
EASING_FUNCTIONS = {
    EasingType.LINEAR: linear,
    EasingType.EASE_IN: ease_in,
    EasingType.EASE_OUT: ease_out,
    EasingType.EASE_IN_OUT: ease_in_out,
    EasingType.EASE_IN_QUAD: ease_in_quad,
    EasingType.EASE_OUT_QUAD: ease_out_quad,
    EasingType.EASE_IN_OUT_QUAD: ease_in_out_quad,
    EasingType.EASE_IN_CUBIC: ease_in_cubic,
    EasingType.EASE_OUT_CUBIC: ease_out_cubic,
    EasingType.EASE_IN_OUT_CUBIC: ease_in_out_cubic,
    EasingType.ELASTIC: elastic,
    EasingType.BOUNCE: bounce,
}


# ============================================================================
# Animation Class
# ============================================================================


@dataclass
class AnimationConfig:
    """Configuration for an animation."""

    duration: float = 0.3  # Duration in seconds
    easing: EasingType = EasingType.EASE_OUT
    fps: int = 60  # Target frames per second
    on_complete: Optional[Callable] = None
    on_update: Optional[Callable[[float], None]] = None


class Animation:
    """
    Base animation class.

    Handles smooth value interpolation over time with easing.

    Attributes:
        start_value (float): Starting value
        end_value (float): Ending value
        config (AnimationConfig): Animation configuration
        start_time (float): Animation start time
        is_running (bool): Whether animation is currently running
    """

    def __init__(
        self, start_value: float, end_value: float, config: Optional[AnimationConfig] = None
    ):
        """
        Initialize animation.

        Args:
            start_value: Starting value
            end_value: Target value
            config: Animation configuration
        """
        self.start_value = start_value
        self.end_value = end_value
        self.config = config or AnimationConfig()
        self.current_value = start_value
        self.start_time = 0.0
        self.is_running = False
        self.animation_id: Optional[str] = None
        self.widget: Optional[tk.Widget] = None

    def start(self, widget: tk.Widget) -> None:
        """
        Start the animation.

        Args:
            widget: Widget to bind animation to (for after() scheduling)
        """
        self.widget = widget
        self.start_time = time.time()
        self.is_running = True
        self._step()

    def stop(self) -> None:
        """Stop the animation."""
        self.is_running = False
        if self.animation_id and self.widget:
            self.widget.after_cancel(self.animation_id)
            self.animation_id = None

    def _step(self) -> None:
        """Perform one animation step."""
        if not self.is_running or not self.widget:
            return

        # Calculate progress
        elapsed = time.time() - self.start_time
        progress = min(elapsed / self.config.duration, 1.0)

        # Apply easing
        easing_func = EASING_FUNCTIONS[self.config.easing]
        eased_progress = easing_func(progress)

        # Interpolate value
        self.current_value = self.start_value + (self.end_value - self.start_value) * eased_progress

        # Callback
        if self.config.on_update:
            self.config.on_update(self.current_value)

        # Check if complete
        if progress >= 1.0:
            self.is_running = False
            if self.config.on_complete:
                self.config.on_complete()
        else:
            # Schedule next frame
            frame_delay = int(1000 / self.config.fps)
            self.animation_id = self.widget.after(frame_delay, self._step)

    def get_value(self) -> float:
        """Get current animated value."""
        return self.current_value


# ============================================================================
# Widget Animators
# ============================================================================


class FadeAnimation:
    """
    Fade in/out animation for widgets.

    Animates widget opacity from 0 (invisible) to 1 (fully visible) or vice versa.
    Note: tkinter doesn't natively support opacity, so this uses alpha blending
    on colors for approximation.
    """

    def __init__(
        self, widget: tk.Widget, duration: float = 0.3, easing: EasingType = EasingType.EASE_IN_OUT
    ):
        """
        Initialize fade animation.

        Args:
            widget: Widget to animate
            duration: Animation duration in seconds
            easing: Easing function type
        """
        self.widget = widget
        self.duration = duration
        self.easing = easing
        self.animation: Optional[Animation] = None

    def fade_in(self, on_complete: Optional[Callable] = None) -> None:
        """
        Fade widget in.

        Args:
            on_complete: Callback when animation completes
        """
        # Show widget first
        if hasattr(self.widget, "grid_info") and not self.widget.grid_info():
            self.widget.grid()

        config = AnimationConfig(
            duration=self.duration,
            easing=self.easing,
            on_complete=on_complete,
            on_update=self._update_alpha,
        )
        self.animation = Animation(0.0, 1.0, config)
        self.animation.start(self.widget)

    def fade_out(self, on_complete: Optional[Callable] = None) -> None:
        """
        Fade widget out.

        Args:
            on_complete: Callback when animation completes
        """

        def hide_widget():
            if hasattr(self.widget, "grid_forget"):
                self.widget.grid_forget()
            if on_complete:
                on_complete()

        config = AnimationConfig(
            duration=self.duration,
            easing=self.easing,
            on_complete=hide_widget,
            on_update=self._update_alpha,
        )
        self.animation = Animation(1.0, 0.0, config)
        self.animation.start(self.widget)

    def _update_alpha(self, alpha: float) -> None:
        """
        Update widget alpha.

        Note: This is a simplified approach. Full opacity requires platform-specific code.
        """
        # For now, we just control visibility state
        # A full implementation would blend colors or use platform-specific APIs
        if alpha < 0.1:
            if hasattr(self.widget, "lower"):
                self.widget.lower()
        else:
            if hasattr(self.widget, "lift"):
                self.widget.lift()


class SlideAnimation:
    """
    Slide animation for widgets.

    Slides widget in/out from specified direction.
    """

    def __init__(
        self, widget: tk.Widget, duration: float = 0.3, easing: EasingType = EasingType.EASE_OUT
    ):
        """
        Initialize slide animation.

        Args:
            widget: Widget to animate
            duration: Animation duration in seconds
            easing: Easing function type
        """
        self.widget = widget
        self.duration = duration
        self.easing = easing
        self.animation: Optional[Animation] = None
        self.original_x = 0
        self.original_y = 0

    def slide_in(
        self, direction: str = "left", distance: int = 100, on_complete: Optional[Callable] = None
    ) -> None:
        """
        Slide widget in from direction.

        Args:
            direction: Direction to slide from ("left", "right", "top", "bottom")
            distance: Distance to slide in pixels
            on_complete: Callback when complete
        """
        # Store original position
        self.original_x = self.widget.winfo_x()
        self.original_y = self.widget.winfo_y()

        # Calculate start position
        if direction == "left":
            start_offset = -distance
            self._animate_x(start_offset, 0, on_complete)
        elif direction == "right":
            start_offset = distance
            self._animate_x(start_offset, 0, on_complete)
        elif direction == "top":
            start_offset = -distance
            self._animate_y(start_offset, 0, on_complete)
        elif direction == "bottom":
            start_offset = distance
            self._animate_y(start_offset, 0, on_complete)

    def slide_out(
        self, direction: str = "left", distance: int = 100, on_complete: Optional[Callable] = None
    ) -> None:
        """
        Slide widget out in direction.

        Args:
            direction: Direction to slide to
            distance: Distance to slide in pixels
            on_complete: Callback when complete
        """
        if direction == "left":
            self._animate_x(0, -distance, on_complete)
        elif direction == "right":
            self._animate_x(0, distance, on_complete)
        elif direction == "top":
            self._animate_y(0, -distance, on_complete)
        elif direction == "bottom":
            self._animate_y(0, distance, on_complete)

    def _animate_x(self, start: float, end: float, on_complete: Optional[Callable]) -> None:
        """Animate X position."""
        config = AnimationConfig(
            duration=self.duration,
            easing=self.easing,
            on_complete=on_complete,
            on_update=lambda x: self.widget.place(x=int(self.original_x + x)),
        )
        self.animation = Animation(start, end, config)
        self.animation.start(self.widget)

    def _animate_y(self, start: float, end: float, on_complete: Optional[Callable]) -> None:
        """Animate Y position."""
        config = AnimationConfig(
            duration=self.duration,
            easing=self.easing,
            on_complete=on_complete,
            on_update=lambda y: self.widget.place(y=int(self.original_y + y)),
        )
        self.animation = Animation(start, end, config)
        self.animation.start(self.widget)


class ScaleAnimation:
    """
    Scale animation for widgets.

    Scales widget size smoothly.
    """

    def __init__(
        self, widget: tk.Widget, duration: float = 0.3, easing: EasingType = EasingType.EASE_OUT
    ):
        """
        Initialize scale animation.

        Args:
            widget: Widget to animate
            duration: Animation duration in seconds
            easing: Easing function type
        """
        self.widget = widget
        self.duration = duration
        self.easing = easing
        self.animation: Optional[Animation] = None
        self.original_width = 0
        self.original_height = 0

    def scale_to(self, scale: float, on_complete: Optional[Callable] = None) -> None:
        """
        Scale widget to specific scale factor.

        Args:
            scale: Target scale (1.0 = original size)
            on_complete: Callback when complete
        """
        self.original_width = self.widget.winfo_width()
        self.original_height = self.widget.winfo_height()

        def update_scale(s: float):
            new_width = int(self.original_width * s)
            new_height = int(self.original_height * s)
            if hasattr(self.widget, "config"):
                self.widget.config(width=new_width, height=new_height)

        config = AnimationConfig(
            duration=self.duration,
            easing=self.easing,
            on_complete=on_complete,
            on_update=update_scale,
        )
        self.animation = Animation(1.0, scale, config)
        self.animation.start(self.widget)

    def pulse(self, scale_max: float = 1.1, cycles: int = 1) -> None:
        """
        Pulse animation (scale up and down).

        Args:
            scale_max: Maximum scale factor
            cycles: Number of pulse cycles
        """
        # Implementation would chain multiple scale animations
        pass


# ============================================================================
# Animation Choreography
# ============================================================================


class AnimationSequence:
    """
    Sequence multiple animations to run one after another.

    Attributes:
        animations (List): List of animation functions to run
        current_index (int): Index of currently running animation
    """

    def __init__(self):
        """Initialize animation sequence."""
        self.animations: List[Callable] = []
        self.current_index = 0
        self.is_running = False

    def add(self, animation_func: Callable) -> "AnimationSequence":
        """
        Add animation to sequence.

        Args:
            animation_func: Function that starts an animation

        Returns:
            Self for chaining
        """
        self.animations.append(animation_func)
        return self

    def start(self) -> None:
        """Start the animation sequence."""
        self.current_index = 0
        self.is_running = True
        self._run_next()

    def stop(self) -> None:
        """Stop the sequence."""
        self.is_running = False

    def _run_next(self) -> None:
        """Run next animation in sequence."""
        if not self.is_running or self.current_index >= len(self.animations):
            self.is_running = False
            return

        # Run current animation with callback to next
        animation = self.animations[self.current_index]
        self.current_index += 1

        # Wrap animation to add our callback
        def next_callback():
            self._run_next()

        animation(on_complete=next_callback)


class AnimationParallel:
    """
    Run multiple animations in parallel.

    Attributes:
        animations (List): List of animation functions
        completed_count (int): Number of completed animations
    """

    def __init__(self):
        """Initialize parallel animations."""
        self.animations: List[Callable] = []
        self.completed_count = 0
        self.on_all_complete: Optional[Callable] = None

    def add(self, animation_func: Callable) -> "AnimationParallel":
        """
        Add animation to parallel group.

        Args:
            animation_func: Function that starts an animation

        Returns:
            Self for chaining
        """
        self.animations.append(animation_func)
        return self

    def start(self, on_complete: Optional[Callable] = None) -> None:
        """
        Start all animations in parallel.

        Args:
            on_complete: Callback when all animations complete
        """
        self.completed_count = 0
        self.on_all_complete = on_complete

        for animation in self.animations:
            animation(on_complete=self._on_animation_complete)

    def _on_animation_complete(self) -> None:
        """Handle individual animation completion."""
        self.completed_count += 1

        if self.completed_count >= len(self.animations):
            if self.on_all_complete:
                self.on_all_complete()


# ============================================================================
# Utility Functions
# ============================================================================


def interpolate(
    start: float, end: float, progress: float, easing: EasingType = EasingType.LINEAR
) -> float:
    """
    Interpolate between two values with easing.

    Args:
        start: Start value
        end: End value
        progress: Progress (0-1)
        easing: Easing function to use

    Returns:
        Interpolated value
    """
    easing_func = EASING_FUNCTIONS[easing]
    eased_progress = easing_func(progress)
    return start + (end - start) * eased_progress


def color_interpolate(
    start_color: str, end_color: str, progress: float, easing: EasingType = EasingType.LINEAR
) -> str:
    """
    Interpolate between two hex colors.

    Args:
        start_color: Start color (hex string)
        end_color: End color (hex string)
        progress: Progress (0-1)
        easing: Easing function

    Returns:
        Interpolated color as hex string
    """
    # Parse colors
    start_rgb = tuple(int(start_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    end_rgb = tuple(int(end_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    # Interpolate each channel
    result_rgb = tuple(
        int(interpolate(start_rgb[i], end_rgb[i], progress, easing)) for i in range(3)
    )

    # Convert back to hex
    return f"#{result_rgb[0]:02x}{result_rgb[1]:02x}{result_rgb[2]:02x}"


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "EasingType",
    "AnimationConfig",
    "Animation",
    "FadeAnimation",
    "SlideAnimation",
    "ScaleAnimation",
    "AnimationSequence",
    "AnimationParallel",
    "interpolate",
    "color_interpolate",
    "EASING_FUNCTIONS",
]
