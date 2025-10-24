"""
FunGen Rewrite - Event Handlers

Event handling and threading support for non-blocking UI updates.
Manages video processing threads, UI callbacks, and inter-thread communication.

Author: ui-architect agent
Date: 2025-10-24
Platform: Cross-platform (Pi + RTX 3090)
"""

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class ProcessingState(Enum):
    """Video processing state enum."""

    IDLE = "idle"
    STARTING = "starting"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProcessingEvent:
    """
    Event data for UI updates.

    Attributes:
        event_type: Type of event (progress, fps, error, etc.)
        data: Event data dictionary
        timestamp: Event timestamp
    """

    event_type: str
    data: Dict[str, Any]
    timestamp: float


class VideoProcessingThread(threading.Thread):
    """
    Background thread for video processing.

    Handles video processing in a separate thread to keep UI responsive.
    Sends progress updates via queue to the main UI thread.

    Attributes:
        video_path (Path): Path to video file or folder
        tracker_type (str): Selected tracker algorithm
        settings (Dict): Processing settings
        event_queue (queue.Queue): Queue for sending events to UI
        stop_event (threading.Event): Event to signal thread stop
    """

    def __init__(
        self,
        video_path: Path,
        tracker_type: str,
        settings: Dict[str, Any],
        event_queue: queue.Queue,
    ):
        """
        Initialize processing thread.

        Args:
            video_path: Path to video file or folder
            tracker_type: Tracker algorithm to use
            settings: Processing settings dictionary
            event_queue: Queue for UI updates
        """
        super().__init__(daemon=True)
        self.video_path = video_path
        self.tracker_type = tracker_type
        self.settings = settings
        self.event_queue = event_queue
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.state = ProcessingState.IDLE

    def run(self) -> None:
        """Main thread execution."""
        try:
            self.state = ProcessingState.STARTING
            self._send_event("state_changed", {"state": self.state.value})

            self.state = ProcessingState.PROCESSING
            self._send_event("state_changed", {"state": self.state.value})

            # Simulate video processing
            # In production, this would call the actual video pipeline
            self._process_video()

            if not self.stop_event.is_set():
                self.state = ProcessingState.COMPLETED
                self._send_event("completed", {"success": True})
            else:
                self.state = ProcessingState.IDLE
                self._send_event("stopped", {})

        except Exception as e:
            self.state = ProcessingState.ERROR
            self._send_event("error", {"message": str(e)})

    def _process_video(self) -> None:
        """
        Simulate video processing (placeholder).

        In production, this would integrate with:
        - core/video_pipeline.py (frame extraction)
        - core/model_manager.py (YOLO detection)
        - trackers/*.py (object tracking)
        - utils/funscript.py (output generation)
        """
        total_frames = 1000
        batch_size = self.settings.get("batch_size", 8)

        for frame_num in range(0, total_frames, batch_size):
            # Check for stop/pause
            if self.stop_event.is_set():
                break

            while self.pause_event.is_set():
                time.sleep(0.1)
                if self.stop_event.is_set():
                    break

            # Simulate processing time
            time.sleep(0.033 * batch_size)  # ~30 FPS

            # Calculate metrics
            progress = (frame_num / total_frames) * 100
            fps = 30.0 + (frame_num % 20) - 10  # Simulate FPS variation
            vram = 5.0 + (frame_num % 100) / 100  # Simulate VRAM usage

            # Send progress update
            self._send_event(
                "progress",
                {
                    "frame": frame_num,
                    "total": total_frames,
                    "progress": progress,
                    "fps": fps,
                    "vram": vram,
                },
            )

    def _send_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Send event to UI thread.

        Args:
            event_type: Type of event
            data: Event data
        """
        event = ProcessingEvent(event_type=event_type, data=data, timestamp=time.time())
        self.event_queue.put(event)

    def stop(self) -> None:
        """Request thread to stop."""
        self.stop_event.set()
        self.pause_event.clear()

    def pause(self) -> None:
        """Pause processing."""
        self.pause_event.set()
        self.state = ProcessingState.PAUSED
        self._send_event("state_changed", {"state": self.state.value})

    def resume(self) -> None:
        """Resume processing."""
        self.pause_event.clear()
        self.state = ProcessingState.PROCESSING
        self._send_event("state_changed", {"state": self.state.value})


class EventHandler:
    """
    Central event handler for UI interactions.

    Manages callbacks, keyboard shortcuts, and event routing.
    """

    def __init__(self):
        """Initialize event handler."""
        self.callbacks: Dict[str, list[Callable]] = {}
        self.processing_thread: Optional[VideoProcessingThread] = None
        self.event_queue: queue.Queue = queue.Queue()

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback for an event type.

        Args:
            event_type: Event type to listen for
            callback: Callback function
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)

    def unregister_callback(self, event_type: str, callback: Callable) -> None:
        """
        Unregister a callback.

        Args:
            event_type: Event type
            callback: Callback function to remove
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].remove(callback)

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit an event to all registered callbacks.

        Args:
            event_type: Event type
            data: Event data
        """
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Error in callback for {event_type}: {e}")

    def start_processing(
        self, video_path: Path, tracker_type: str, settings: Dict[str, Any]
    ) -> bool:
        """
        Start video processing thread.

        Args:
            video_path: Path to video file/folder
            tracker_type: Tracker algorithm
            settings: Processing settings

        Returns:
            True if started successfully, False otherwise
        """
        if self.processing_thread and self.processing_thread.is_alive():
            return False

        self.processing_thread = VideoProcessingThread(
            video_path=video_path,
            tracker_type=tracker_type,
            settings=settings,
            event_queue=self.event_queue,
        )
        self.processing_thread.start()
        return True

    def stop_processing(self) -> None:
        """Stop processing thread."""
        if self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread.join(timeout=2.0)

    def pause_processing(self) -> None:
        """Pause processing thread."""
        if self.processing_thread:
            self.processing_thread.pause()

    def resume_processing(self) -> None:
        """Resume processing thread."""
        if self.processing_thread:
            self.processing_thread.resume()

    def process_events(self) -> None:
        """
        Process pending events from the event queue.

        Should be called regularly from the UI thread.
        """
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                self.emit_event(event.event_type, event.data)
            except queue.Empty:
                break


class KeyboardShortcuts:
    """
    Keyboard shortcut handler.

    Manages keyboard shortcuts for the UI.
    """

    # Default shortcuts
    SHORTCUTS = {
        "<Control-o>": "open_video",
        "<Control-O>": "open_folder",
        "<Control-s>": "save",
        "<Control-comma>": "settings",
        "<Control-q>": "quit",
        "<space>": "play_pause",
        "<Escape>": "stop",
        "<F1>": "help",
        "<F5>": "refresh",
    }

    @staticmethod
    def bind_shortcuts(widget, callback_map: Dict[str, Callable]) -> None:
        """
        Bind keyboard shortcuts to a widget.

        Args:
            widget: Tkinter widget to bind to
            callback_map: Map of action names to callback functions
        """
        for key, action in KeyboardShortcuts.SHORTCUTS.items():
            if action in callback_map:
                widget.bind(key, lambda e, a=action: callback_map[a]())


class ProgressTracker:
    """
    Track processing progress and calculate statistics.

    Calculates:
    - Average FPS
    - Processing time
    - Estimated time remaining
    - VRAM usage trends
    """

    def __init__(self):
        """Initialize progress tracker."""
        self.start_time: Optional[float] = None
        self.frames_processed: int = 0
        self.total_frames: int = 0
        self.fps_history: list[float] = []
        self.max_history_size = 30

    def start(self, total_frames: int) -> None:
        """
        Start tracking.

        Args:
            total_frames: Total number of frames to process
        """
        self.start_time = time.time()
        self.frames_processed = 0
        self.total_frames = total_frames
        self.fps_history = []

    def update(self, frames_processed: int, current_fps: float) -> None:
        """
        Update progress.

        Args:
            frames_processed: Number of frames processed so far
            current_fps: Current FPS
        """
        self.frames_processed = frames_processed
        self.fps_history.append(current_fps)

        # Keep only recent history
        if len(self.fps_history) > self.max_history_size:
            self.fps_history.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.start_time:
            return {}

        elapsed = time.time() - self.start_time
        progress = (self.frames_processed / self.total_frames * 100) if self.total_frames > 0 else 0
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

        # Estimate time remaining
        if avg_fps > 0 and self.frames_processed > 0:
            frames_remaining = self.total_frames - self.frames_processed
            eta_seconds = frames_remaining / avg_fps
        else:
            eta_seconds = 0

        return {
            "elapsed": elapsed,
            "progress": progress,
            "frames_processed": self.frames_processed,
            "total_frames": self.total_frames,
            "avg_fps": avg_fps,
            "eta_seconds": eta_seconds,
        }


def format_time(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_percentage(value: float) -> str:
    """
    Format percentage with 1 decimal place.

    Args:
        value: Percentage value (0-100)

    Returns:
        Formatted string
    """
    return f"{value:.1f}%"
