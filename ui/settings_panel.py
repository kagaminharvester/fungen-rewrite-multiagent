"""
FunGen Rewrite - Settings Panel

Configuration UI for tracker selection, performance settings, and preferences.

Author: ui-architect agent
Date: 2025-10-24
Platform: Cross-platform (Pi + RTX 3090)
"""

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, Optional


class SettingsPanel(tk.Toplevel):
    """
    Settings panel window.

    Features:
        - Tracker algorithm selection
        - Batch size configuration
        - Hardware acceleration toggle
        - TensorRT FP16 optimization
        - Model directory selection
        - Output directory selection
        - VR format detection settings
        - Theme selection (light/dark)
        - Save/load configuration profiles

    Attributes:
        settings (Dict): Current settings dictionary
        config_file (Path): Path to configuration file
    """

    def __init__(self, parent: tk.Widget, config_file: str = "/home/pi/elo_elo_320/config.json"):
        """
        Initialize settings panel.

        Args:
            parent: Parent window
            config_file: Path to configuration JSON file
        """
        super().__init__(parent)

        self.title("FunGen Settings")
        self.geometry("700x600")
        self.resizable(True, True)

        self.config_file = Path(config_file)
        self.settings: Dict[str, Any] = {}

        # Load existing settings
        self._load_settings()

        # Create UI
        self._create_widgets()

        # Center window
        self.transient(parent)
        self.grab_set()

    def _load_settings(self) -> None:
        """Load settings from config file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    self.settings = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.settings = self._get_default_settings()
        else:
            self.settings = self._get_default_settings()

    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings."""
        return {
            "tracker": {
                "algorithm": "hybrid",
                "iou_threshold": 0.5,
                "confidence_threshold": 0.3,
                "max_age": 30,
                "enable_reid": True,
            },
            "processing": {
                "batch_size": 8,
                "hw_accel": True,
                "tensorrt_fp16": True,
                "num_workers": "auto",
            },
            "paths": {
                "model_dir": "/home/pi/elo_elo_320/models",
                "output_dir": "/home/pi/elo_elo_320/output",
            },
            "vr": {"auto_detect": True, "default_format": "sbs_fisheye"},
            "ui": {"theme": "dark", "auto_refresh_agents": True, "show_video_preview": True},
        }

    def _save_settings(self) -> None:
        """Save settings to config file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(self.settings, f, indent=2)
            messagebox.showinfo("Success", "Settings saved successfully!")
        except IOError as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def _create_widgets(self) -> None:
        """Create settings UI."""
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Create tabs
        self._create_tracker_tab(notebook)
        self._create_processing_tab(notebook)
        self._create_paths_tab(notebook)
        self._create_vr_tab(notebook)
        self._create_ui_tab(notebook)

        # Bottom buttons
        self._create_button_panel()

    def _create_tracker_tab(self, notebook: ttk.Notebook) -> None:
        """Create tracker settings tab."""
        frame = ttk.Frame(notebook, padding="20")
        notebook.add(frame, text="Tracker")

        # Algorithm selection
        ttk.Label(frame, text="Tracker Algorithm:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        self.tracker_var = tk.StringVar(value=self.settings["tracker"]["algorithm"])
        algorithms = [
            ("ByteTrack (Fast)", "bytetrack"),
            ("BoT-SORT (Accurate)", "botsort"),
            ("Hybrid (Recommended)", "hybrid"),
        ]

        for idx, (label, value) in enumerate(algorithms):
            ttk.Radiobutton(frame, text=label, variable=self.tracker_var, value=value).grid(
                row=idx + 1, column=0, sticky="w", padx=20
            )

        # IoU threshold
        ttk.Label(frame, text="IoU Threshold:", font=("Arial", 10, "bold")).grid(
            row=5, column=0, sticky="w", pady=(20, 5)
        )
        self.iou_var = tk.DoubleVar(value=self.settings["tracker"]["iou_threshold"])
        ttk.Scale(
            frame, from_=0.1, to=0.9, variable=self.iou_var, orient="horizontal", length=300
        ).grid(row=6, column=0, sticky="w", padx=20)
        ttk.Label(frame, textvariable=self.iou_var).grid(row=6, column=1, padx=10)

        # Confidence threshold
        ttk.Label(frame, text="Confidence Threshold:", font=("Arial", 10, "bold")).grid(
            row=7, column=0, sticky="w", pady=(20, 5)
        )
        self.conf_var = tk.DoubleVar(value=self.settings["tracker"]["confidence_threshold"])
        ttk.Scale(
            frame, from_=0.1, to=0.9, variable=self.conf_var, orient="horizontal", length=300
        ).grid(row=8, column=0, sticky="w", padx=20)
        ttk.Label(frame, textvariable=self.conf_var).grid(row=8, column=1, padx=10)

        # Max age
        ttk.Label(frame, text="Max Track Age (frames):", font=("Arial", 10, "bold")).grid(
            row=9, column=0, sticky="w", pady=(20, 5)
        )
        self.max_age_var = tk.IntVar(value=self.settings["tracker"]["max_age"])
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.max_age_var, width=10).grid(
            row=10, column=0, sticky="w", padx=20
        )

        # ReID toggle
        self.reid_var = tk.BooleanVar(value=self.settings["tracker"]["enable_reid"])
        ttk.Checkbutton(frame, text="Enable Re-Identification (ReID)", variable=self.reid_var).grid(
            row=11, column=0, sticky="w", pady=(20, 0)
        )

    def _create_processing_tab(self, notebook: ttk.Notebook) -> None:
        """Create processing settings tab."""
        frame = ttk.Frame(notebook, padding="20")
        notebook.add(frame, text="Processing")

        # Batch size
        ttk.Label(frame, text="Batch Size:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        self.batch_size_var = tk.IntVar(value=self.settings["processing"]["batch_size"])
        ttk.Spinbox(frame, from_=1, to=32, textvariable=self.batch_size_var, width=10).grid(
            row=1, column=0, sticky="w", padx=20
        )

        ttk.Label(
            frame,
            text="Higher = better GPU utilization, more VRAM",
            foreground="gray",
            font=("Arial", 9, "italic"),
        ).grid(row=2, column=0, sticky="w", padx=20, pady=(0, 20))

        # Hardware acceleration
        self.hw_accel_var = tk.BooleanVar(value=self.settings["processing"]["hw_accel"])
        ttk.Checkbutton(
            frame, text="Hardware Acceleration (NVDEC/CUDA)", variable=self.hw_accel_var
        ).grid(row=3, column=0, sticky="w", pady=5)

        # TensorRT FP16
        self.tensorrt_var = tk.BooleanVar(value=self.settings["processing"]["tensorrt_fp16"])
        ttk.Checkbutton(
            frame, text="TensorRT FP16 Optimization (40% speedup)", variable=self.tensorrt_var
        ).grid(row=4, column=0, sticky="w", pady=5)

        # Number of workers
        ttk.Label(frame, text="Parallel Workers:", font=("Arial", 10, "bold")).grid(
            row=5, column=0, sticky="w", pady=(20, 5)
        )
        self.workers_var = tk.StringVar(value=str(self.settings["processing"]["num_workers"]))
        worker_frame = ttk.Frame(frame)
        worker_frame.grid(row=6, column=0, sticky="w", padx=20)

        ttk.Radiobutton(
            worker_frame, text="Auto-detect", variable=self.workers_var, value="auto"
        ).pack(side="left", padx=5)

        ttk.Radiobutton(
            worker_frame, text="Manual:", variable=self.workers_var, value="manual"
        ).pack(side="left", padx=5)

        self.workers_manual_var = tk.IntVar(value=4)
        ttk.Spinbox(
            worker_frame, from_=1, to=12, textvariable=self.workers_manual_var, width=5
        ).pack(side="left")

    def _create_paths_tab(self, notebook: ttk.Notebook) -> None:
        """Create paths settings tab."""
        frame = ttk.Frame(notebook, padding="20")
        notebook.add(frame, text="Paths")

        # Model directory
        ttk.Label(frame, text="Model Directory:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        self.model_dir_var = tk.StringVar(value=self.settings["paths"]["model_dir"])
        ttk.Entry(frame, textvariable=self.model_dir_var, width=50).grid(
            row=1, column=0, sticky="w", padx=20
        )
        ttk.Button(
            frame, text="Browse...", command=lambda: self._browse_directory(self.model_dir_var)
        ).grid(row=1, column=1, padx=5)

        # Output directory
        ttk.Label(frame, text="Output Directory:", font=("Arial", 10, "bold")).grid(
            row=2, column=0, sticky="w", pady=(20, 5)
        )
        self.output_dir_var = tk.StringVar(value=self.settings["paths"]["output_dir"])
        ttk.Entry(frame, textvariable=self.output_dir_var, width=50).grid(
            row=3, column=0, sticky="w", padx=20
        )
        ttk.Button(
            frame, text="Browse...", command=lambda: self._browse_directory(self.output_dir_var)
        ).grid(row=3, column=1, padx=5)

    def _create_vr_tab(self, notebook: ttk.Notebook) -> None:
        """Create VR settings tab."""
        frame = ttk.Frame(notebook, padding="20")
        notebook.add(frame, text="VR Settings")

        # Auto-detect
        self.vr_auto_var = tk.BooleanVar(value=self.settings["vr"]["auto_detect"])
        ttk.Checkbutton(
            frame, text="Auto-detect VR format from filename", variable=self.vr_auto_var
        ).grid(row=0, column=0, sticky="w", pady=10)

        # Default format
        ttk.Label(frame, text="Default VR Format:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky="w", pady=(20, 10)
        )

        self.vr_format_var = tk.StringVar(value=self.settings["vr"]["default_format"])
        formats = [
            ("SBS Fisheye 180°", "sbs_fisheye"),
            ("SBS Equirectangular 180°", "sbs_equirect"),
            ("Mono (2D)", "mono"),
        ]

        for idx, (label, value) in enumerate(formats):
            ttk.Radiobutton(frame, text=label, variable=self.vr_format_var, value=value).grid(
                row=idx + 2, column=0, sticky="w", padx=20
            )

    def _create_ui_tab(self, notebook: ttk.Notebook) -> None:
        """Create UI settings tab."""
        frame = ttk.Frame(notebook, padding="20")
        notebook.add(frame, text="UI")

        # Theme
        ttk.Label(frame, text="Theme:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        self.theme_var = tk.StringVar(value=self.settings["ui"]["theme"])
        themes = [("Dark", "dark"), ("Light", "light")]

        for idx, (label, value) in enumerate(themes):
            ttk.Radiobutton(frame, text=label, variable=self.theme_var, value=value).grid(
                row=idx + 1, column=0, sticky="w", padx=20
            )

        # Auto-refresh agents
        self.auto_refresh_var = tk.BooleanVar(value=self.settings["ui"]["auto_refresh_agents"])
        ttk.Checkbutton(
            frame, text="Auto-refresh agent dashboard (2s)", variable=self.auto_refresh_var
        ).grid(row=3, column=0, sticky="w", pady=(20, 5))

        # Show video preview
        self.preview_var = tk.BooleanVar(value=self.settings["ui"]["show_video_preview"])
        ttk.Checkbutton(frame, text="Show video preview", variable=self.preview_var).grid(
            row=4, column=0, sticky="w", pady=5
        )

    def _create_button_panel(self) -> None:
        """Create bottom button panel."""
        button_frame = ttk.Frame(self, padding="10")
        button_frame.pack(side="bottom", fill="x")

        ttk.Button(button_frame, text="Save", command=self._on_save, width=15).pack(
            side="right", padx=5
        )

        ttk.Button(button_frame, text="Cancel", command=self.destroy, width=15).pack(
            side="right", padx=5
        )

        ttk.Button(button_frame, text="Reset to Defaults", command=self._on_reset).pack(
            side="left", padx=5
        )

    def _browse_directory(self, var: tk.StringVar) -> None:
        """
        Browse for directory.

        Args:
            var: StringVar to update with selected path
        """
        directory = filedialog.askdirectory(initialdir=var.get())
        if directory:
            var.set(directory)

    def _on_save(self) -> None:
        """Save settings and close."""
        # Update settings dictionary
        self.settings["tracker"]["algorithm"] = self.tracker_var.get()
        self.settings["tracker"]["iou_threshold"] = self.iou_var.get()
        self.settings["tracker"]["confidence_threshold"] = self.conf_var.get()
        self.settings["tracker"]["max_age"] = self.max_age_var.get()
        self.settings["tracker"]["enable_reid"] = self.reid_var.get()

        self.settings["processing"]["batch_size"] = self.batch_size_var.get()
        self.settings["processing"]["hw_accel"] = self.hw_accel_var.get()
        self.settings["processing"]["tensorrt_fp16"] = self.tensorrt_var.get()
        self.settings["processing"]["num_workers"] = (
            self.workers_var.get()
            if self.workers_var.get() == "auto"
            else self.workers_manual_var.get()
        )

        self.settings["paths"]["model_dir"] = self.model_dir_var.get()
        self.settings["paths"]["output_dir"] = self.output_dir_var.get()

        self.settings["vr"]["auto_detect"] = self.vr_auto_var.get()
        self.settings["vr"]["default_format"] = self.vr_format_var.get()

        self.settings["ui"]["theme"] = self.theme_var.get()
        self.settings["ui"]["auto_refresh_agents"] = self.auto_refresh_var.get()
        self.settings["ui"]["show_video_preview"] = self.preview_var.get()

        # Save to file
        self._save_settings()
        self.destroy()

    def _on_reset(self) -> None:
        """Reset to default settings."""
        if messagebox.askyesno("Reset Settings", "Reset all settings to defaults?"):
            self.settings = self._get_default_settings()
            self.destroy()
            # Reopen with defaults
            SettingsPanel(self.master, str(self.config_file))


def main():
    """Test the settings panel."""
    root = tk.Tk()
    root.withdraw()

    panel = SettingsPanel(root)
    root.wait_window(panel)
    root.destroy()


if __name__ == "__main__":
    main()
