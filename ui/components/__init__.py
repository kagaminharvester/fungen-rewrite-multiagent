"""
FunGen Rewrite - Reusable UI Components

Collection of reusable tkinter widgets and utilities.

Components:
    - ProgressBar: Enhanced progress bar with percentage display
    - Tooltip: Hover tooltip for buttons and labels
    - FileDialog: Custom file/folder selection dialog
    - StatusBar: Status bar with icon support
    - MetricDisplay: Real-time metric display (FPS, VRAM, etc.)

Author: ui-architect agent
Date: 2025-10-24
"""

__version__ = "1.0.0"
__all__ = ["ProgressBar", "Tooltip", "FileDialog", "StatusBar", "MetricDisplay"]

from ui.components.metric_display import MetricDisplay
from ui.components.progress_bar import ProgressBar
from ui.components.status_bar import StatusBar
from ui.components.tooltip import Tooltip
