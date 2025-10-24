"""
FunGen Rewrite - Theme Management System

Comprehensive theme system supporting:
- Multiple built-in themes (dark, light, high-contrast, custom)
- Dynamic theme switching without restart
- Color palette management
- Widget style configuration
- Theme persistence
- Accessibility features

Author: ui-enhancer agent
Date: 2025-10-24
Platform: Cross-platform (Pi + RTX 3090)
"""

import json
import tkinter as tk
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, Optional, Tuple

# ============================================================================
# Theme Data Structures
# ============================================================================


class ThemeType(Enum):
    """Available theme types."""

    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"
    NORD = "nord"
    DRACULA = "dracula"
    CUSTOM = "custom"


@dataclass
class ColorPalette:
    """
    Color palette for a theme.

    Attributes:
        primary: Primary accent color
        secondary: Secondary accent color
        background: Main background color
        surface: Surface/card background color
        text_primary: Primary text color
        text_secondary: Secondary text color
        success: Success color
        warning: Warning color
        error: Error color
        info: Info color
    """

    primary: str
    secondary: str
    background: str
    surface: str
    text_primary: str
    text_secondary: str
    success: str
    warning: str
    error: str
    info: str
    border: str = "#555555"
    hover: str = "#404040"
    active: str = "#505050"
    disabled: str = "#808080"


@dataclass
class Theme:
    """
    Complete theme configuration.

    Attributes:
        name: Theme name
        type: Theme type
        colors: Color palette
        fonts: Font configuration
        widget_styles: Custom widget styles
    """

    name: str
    type: ThemeType
    colors: ColorPalette
    fonts: Dict[str, Tuple[str, int, str]] = field(default_factory=dict)
    widget_styles: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ============================================================================
# Built-in Theme Definitions
# ============================================================================

DARK_THEME = Theme(
    name="Dark",
    type=ThemeType.DARK,
    colors=ColorPalette(
        primary="#3498db",
        secondary="#9b59b6",
        background="#1e1e1e",
        surface="#2b2b2b",
        text_primary="#ffffff",
        text_secondary="#b0b0b0",
        success="#2ecc71",
        warning="#f39c12",
        error="#e74c3c",
        info="#3498db",
        border="#444444",
        hover="#353535",
        active="#404040",
    ),
    fonts={
        "default": ("Arial", 9, "normal"),
        "heading": ("Arial", 12, "bold"),
        "mono": ("Courier New", 9, "normal"),
        "large": ("Arial", 14, "bold"),
    },
)

LIGHT_THEME = Theme(
    name="Light",
    type=ThemeType.LIGHT,
    colors=ColorPalette(
        primary="#2980b9",
        secondary="#8e44ad",
        background="#f5f5f5",
        surface="#ffffff",
        text_primary="#2c3e50",
        text_secondary="#7f8c8d",
        success="#27ae60",
        warning="#d68910",
        error="#c0392b",
        info="#2980b9",
        border="#cccccc",
        hover="#e8e8e8",
        active="#d0d0d0",
    ),
    fonts={
        "default": ("Arial", 9, "normal"),
        "heading": ("Arial", 12, "bold"),
        "mono": ("Courier New", 9, "normal"),
        "large": ("Arial", 14, "bold"),
    },
)

HIGH_CONTRAST_THEME = Theme(
    name="High Contrast",
    type=ThemeType.HIGH_CONTRAST,
    colors=ColorPalette(
        primary="#00ffff",
        secondary="#ff00ff",
        background="#000000",
        surface="#1a1a1a",
        text_primary="#ffffff",
        text_secondary="#cccccc",
        success="#00ff00",
        warning="#ffff00",
        error="#ff0000",
        info="#00ffff",
        border="#ffffff",
        hover="#333333",
        active="#4d4d4d",
    ),
    fonts={
        "default": ("Arial", 10, "bold"),
        "heading": ("Arial", 14, "bold"),
        "mono": ("Courier New", 10, "bold"),
        "large": ("Arial", 16, "bold"),
    },
)

NORD_THEME = Theme(
    name="Nord",
    type=ThemeType.NORD,
    colors=ColorPalette(
        primary="#88c0d0",
        secondary="#b48ead",
        background="#2e3440",
        surface="#3b4252",
        text_primary="#eceff4",
        text_secondary="#d8dee9",
        success="#a3be8c",
        warning="#ebcb8b",
        error="#bf616a",
        info="#88c0d0",
        border="#4c566a",
        hover="#434c5e",
        active="#4c566a",
    ),
    fonts={
        "default": ("Arial", 9, "normal"),
        "heading": ("Arial", 12, "bold"),
        "mono": ("Fira Code", 9, "normal"),
        "large": ("Arial", 14, "bold"),
    },
)

DRACULA_THEME = Theme(
    name="Dracula",
    type=ThemeType.DRACULA,
    colors=ColorPalette(
        primary="#bd93f9",
        secondary="#ff79c6",
        background="#282a36",
        surface="#383a59",
        text_primary="#f8f8f2",
        text_secondary="#6272a4",
        success="#50fa7b",
        warning="#f1fa8c",
        error="#ff5555",
        info="#8be9fd",
        border="#44475a",
        hover="#44475a",
        active="#6272a4",
    ),
    fonts={
        "default": ("Arial", 9, "normal"),
        "heading": ("Arial", 12, "bold"),
        "mono": ("Fira Code", 9, "normal"),
        "large": ("Arial", 14, "bold"),
    },
)

# Theme registry
BUILTIN_THEMES: Dict[ThemeType, Theme] = {
    ThemeType.DARK: DARK_THEME,
    ThemeType.LIGHT: LIGHT_THEME,
    ThemeType.HIGH_CONTRAST: HIGH_CONTRAST_THEME,
    ThemeType.NORD: NORD_THEME,
    ThemeType.DRACULA: DRACULA_THEME,
}


# ============================================================================
# Theme Manager
# ============================================================================


class ThemeManager:
    """
    Central theme management system.

    Features:
        - Load and apply themes
        - Dynamic theme switching
        - Custom theme creation
        - Theme persistence
        - Widget style management
        - Event callbacks for theme changes

    Attributes:
        current_theme (Theme): Currently active theme
        custom_themes (Dict[str, Theme]): User-defined custom themes
        callbacks (List[Callable]): Theme change callbacks
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize theme manager.

        Args:
            config_dir: Directory for theme configuration files
        """
        self.config_dir = config_dir or Path.home() / ".fungen" / "themes"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.current_theme: Theme = DARK_THEME
        self.custom_themes: Dict[str, Theme] = {}
        self.callbacks: list = []

        # Try to import sv_ttk for enhanced theming
        self.sv_ttk_available = False
        try:
            import sv_ttk

            self.sv_ttk = sv_ttk
            self.sv_ttk_available = True
        except ImportError:
            pass

        # Load custom themes
        self._load_custom_themes()

        # Load last used theme
        self._load_last_theme()

    def get_available_themes(self) -> Dict[str, Theme]:
        """
        Get all available themes (built-in + custom).

        Returns:
            Dictionary of theme name to Theme object
        """
        themes = {theme.name: theme for theme in BUILTIN_THEMES.values()}
        themes.update(self.custom_themes)
        return themes

    def apply_theme(self, theme: Theme, root: Optional[tk.Tk] = None) -> None:
        """
        Apply theme to the application.

        Args:
            theme: Theme to apply
            root: Root window (if provided, will configure ttk styles)
        """
        self.current_theme = theme

        # Apply sv_ttk theme if available
        if self.sv_ttk_available and root:
            # Map our theme types to sv_ttk themes
            if theme.type == ThemeType.LIGHT:
                self.sv_ttk.set_theme("light")
            else:
                self.sv_ttk.set_theme("dark")

        # Configure ttk styles
        if root:
            self._configure_ttk_styles(root)

        # Save as last used theme
        self._save_last_theme(theme.name)

        # Notify callbacks
        for callback in self.callbacks:
            callback(theme)

    def switch_theme(self, theme_name: str, root: Optional[tk.Tk] = None) -> None:
        """
        Switch to a theme by name.

        Args:
            theme_name: Name of theme to switch to
            root: Root window

        Raises:
            ValueError: If theme not found
        """
        themes = self.get_available_themes()
        if theme_name not in themes:
            raise ValueError(f"Theme '{theme_name}' not found")

        self.apply_theme(themes[theme_name], root)

    def toggle_dark_light(self, root: Optional[tk.Tk] = None) -> None:
        """
        Toggle between dark and light themes.

        Args:
            root: Root window
        """
        if self.current_theme.type == ThemeType.LIGHT:
            self.apply_theme(DARK_THEME, root)
        else:
            self.apply_theme(LIGHT_THEME, root)

    def create_custom_theme(
        self,
        name: str,
        base_theme: ThemeType = ThemeType.DARK,
        color_overrides: Optional[Dict[str, str]] = None,
    ) -> Theme:
        """
        Create a custom theme based on a built-in theme.

        Args:
            name: Custom theme name
            base_theme: Base theme to start from
            color_overrides: Dictionary of color overrides

        Returns:
            New custom theme
        """
        base = BUILTIN_THEMES[base_theme]

        # Create color palette with overrides
        colors_dict = {
            "primary": base.colors.primary,
            "secondary": base.colors.secondary,
            "background": base.colors.background,
            "surface": base.colors.surface,
            "text_primary": base.colors.text_primary,
            "text_secondary": base.colors.text_secondary,
            "success": base.colors.success,
            "warning": base.colors.warning,
            "error": base.colors.error,
            "info": base.colors.info,
            "border": base.colors.border,
            "hover": base.colors.hover,
            "active": base.colors.active,
        }

        if color_overrides:
            colors_dict.update(color_overrides)

        custom_theme = Theme(
            name=name,
            type=ThemeType.CUSTOM,
            colors=ColorPalette(**colors_dict),
            fonts=base.fonts.copy(),
        )

        # Save custom theme
        self.custom_themes[name] = custom_theme
        self._save_custom_theme(custom_theme)

        return custom_theme

    def register_callback(self, callback: callable) -> None:
        """
        Register a callback for theme changes.

        Args:
            callback: Function to call when theme changes (receives Theme)
        """
        self.callbacks.append(callback)

    def _configure_ttk_styles(self, root: tk.Tk) -> None:
        """
        Configure ttk widget styles based on current theme.

        Args:
            root: Root window
        """
        style = ttk.Style(root)
        colors = self.current_theme.colors

        # Configure common styles
        style.configure(".", background=colors.surface, foreground=colors.text_primary)
        style.configure("TFrame", background=colors.background)
        style.configure("TLabel", background=colors.surface, foreground=colors.text_primary)
        style.configure("TButton", background=colors.primary, foreground=colors.text_primary)
        style.map("TButton", background=[("active", colors.active), ("disabled", colors.disabled)])

        # Configure specific widget styles
        style.configure("TEntry", fieldbackground=colors.surface, foreground=colors.text_primary)
        style.configure("TCombobox", fieldbackground=colors.surface, foreground=colors.text_primary)

        # Progress bar
        style.configure("TProgressbar", background=colors.primary, troughcolor=colors.surface)

        # Notebook
        style.configure("TNotebook", background=colors.background)
        style.configure("TNotebook.Tab", background=colors.surface, foreground=colors.text_primary)
        style.map(
            "TNotebook.Tab",
            background=[("selected", colors.primary)],
            foreground=[("selected", colors.text_primary)],
        )

        # Scrollbar
        style.configure("TScrollbar", background=colors.surface, troughcolor=colors.background)

    def _load_custom_themes(self) -> None:
        """Load custom themes from config directory."""
        themes_file = self.config_dir / "custom_themes.json"
        if themes_file.exists():
            try:
                with open(themes_file, "r") as f:
                    data = json.load(f)

                for theme_data in data.get("themes", []):
                    theme = self._theme_from_dict(theme_data)
                    self.custom_themes[theme.name] = theme
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading custom themes: {e}")

    def _save_custom_theme(self, theme: Theme) -> None:
        """Save a custom theme to config file."""
        themes_file = self.config_dir / "custom_themes.json"

        # Load existing
        themes = []
        if themes_file.exists():
            try:
                with open(themes_file, "r") as f:
                    data = json.load(f)
                    themes = data.get("themes", [])
            except json.JSONDecodeError:
                pass

        # Update or add
        theme_dict = self._theme_to_dict(theme)
        existing = next((i for i, t in enumerate(themes) if t["name"] == theme.name), None)
        if existing is not None:
            themes[existing] = theme_dict
        else:
            themes.append(theme_dict)

        # Save
        with open(themes_file, "w") as f:
            json.dump({"themes": themes}, f, indent=2)

    def _load_last_theme(self) -> None:
        """Load the last used theme."""
        config_file = self.config_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                    theme_name = data.get("last_theme")

                    if theme_name:
                        themes = self.get_available_themes()
                        if theme_name in themes:
                            self.current_theme = themes[theme_name]
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_last_theme(self, theme_name: str) -> None:
        """Save the last used theme name."""
        config_file = self.config_dir / "config.json"
        data = {"last_theme": theme_name}

        with open(config_file, "w") as f:
            json.dump(data, f, indent=2)

    def _theme_to_dict(self, theme: Theme) -> Dict[str, Any]:
        """Convert theme to dictionary."""
        return {
            "name": theme.name,
            "type": theme.type.value,
            "colors": {
                "primary": theme.colors.primary,
                "secondary": theme.colors.secondary,
                "background": theme.colors.background,
                "surface": theme.colors.surface,
                "text_primary": theme.colors.text_primary,
                "text_secondary": theme.colors.text_secondary,
                "success": theme.colors.success,
                "warning": theme.colors.warning,
                "error": theme.colors.error,
                "info": theme.colors.info,
                "border": theme.colors.border,
                "hover": theme.colors.hover,
                "active": theme.colors.active,
            },
            "fonts": theme.fonts,
        }

    def _theme_from_dict(self, data: Dict[str, Any]) -> Theme:
        """Create theme from dictionary."""
        return Theme(
            name=data["name"],
            type=ThemeType(data["type"]),
            colors=ColorPalette(**data["colors"]),
            fonts=data.get("fonts", {}),
        )


# ============================================================================
# Convenience Functions
# ============================================================================

# Global theme manager instance
_theme_manager: Optional[ThemeManager] = None


def get_theme_manager() -> ThemeManager:
    """
    Get the global theme manager instance.

    Returns:
        Global ThemeManager instance
    """
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager


def apply_theme(theme_name: str, root: Optional[tk.Tk] = None) -> None:
    """
    Convenience function to apply a theme.

    Args:
        theme_name: Name of theme to apply
        root: Root window
    """
    manager = get_theme_manager()
    manager.switch_theme(theme_name, root)


def toggle_theme(root: Optional[tk.Tk] = None) -> None:
    """
    Convenience function to toggle between dark and light.

    Args:
        root: Root window
    """
    manager = get_theme_manager()
    manager.toggle_dark_light(root)


def get_current_colors() -> ColorPalette:
    """
    Get the current theme's color palette.

    Returns:
        Current color palette
    """
    manager = get_theme_manager()
    return manager.current_theme.colors


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "ThemeType",
    "ColorPalette",
    "Theme",
    "ThemeManager",
    "get_theme_manager",
    "apply_theme",
    "toggle_theme",
    "get_current_colors",
    "DARK_THEME",
    "LIGHT_THEME",
    "HIGH_CONTRAST_THEME",
    "NORD_THEME",
    "DRACULA_THEME",
]
