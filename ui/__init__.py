"""
FunGen Rewrite - UI Module

Modern UI implementation with tkinter + sv_ttk theme, featuring real-time
agent progress visualization, video preview, and performance monitoring.

Modules:
    - main_window: Primary application window with video controls
    - agent_dashboard: Real-time visualization of 15 agent progress bars
    - settings_panel: Configuration UI for tracker selection and settings
    - event_handlers: UI event handling and threading support
    - components: Reusable UI widgets (progress bars, tooltips, etc.)

Author: ui-architect agent
Date: 2025-10-24
Platform: Raspberry Pi (dev) + RTX 3090 (prod)
"""

__version__ = "1.0.0"
__all__ = ["MainWindow", "AgentDashboard", "SettingsPanel"]

# Import main classes for easy access
try:
    from ui.main_window import MainWindow
except ImportError:
    MainWindow = None

try:
    from ui.agent_dashboard import AgentDashboard
except ImportError:
    AgentDashboard = None

try:
    from ui.settings_panel import SettingsPanel
except ImportError:
    SettingsPanel = None
