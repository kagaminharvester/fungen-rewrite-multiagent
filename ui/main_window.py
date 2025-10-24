"""
FunGen Rewrite - Main Window

Primary application window with tkinter + sv_ttk theme, featuring:
- Video selection and preview
- Agent dashboard integration
- Settings panel access
- Real-time FPS/VRAM monitoring
- Start/stop/pause controls

Author: ui-architect agent
Date: 2025-10-24
Platform: Cross-platform (Pi + RTX 3090)
"""

import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, Optional

# Try to import sv_ttk for modern theme, fallback to default
try:
    import sv_ttk

    SV_TTK_AVAILABLE = True
except ImportError:
    SV_TTK_AVAILABLE = False
    print("Warning: sv_ttk not available, using default theme")


class MainWindow(tk.Tk):
    """
    Main application window for FunGen Rewrite.

    Features:
        - Modern UI with sv_ttk theme (light/dark modes)
        - Video file/folder selection
        - Tracker algorithm dropdown
        - Real-time agent progress dashboard
        - FPS/VRAM status display
        - Video preview area
        - Settings panel integration
        - Non-blocking threading for processing

    Attributes:
        video_path (Optional[Path]): Currently selected video file/folder
        tracker_type (str): Selected tracker algorithm
        is_processing (bool): Whether video processing is active
        update_queue (queue.Queue): Thread-safe queue for UI updates
    """

    def __init__(self):
        """Initialize the main window with all UI components."""
        super().__init__()

        # Window configuration
        self.title("FunGen Rewrite - AI-Powered Funscript Generator")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        # Apply modern theme
        if SV_TTK_AVAILABLE:
            sv_ttk.set_theme("dark")  # Default to dark theme

        # State variables
        self.video_path: Optional[Path] = None
        self.tracker_type: str = "hybrid"
        self.is_processing: bool = False
        self.update_queue: queue.Queue = queue.Queue()
        self.processing_thread: Optional[threading.Thread] = None

        # Performance metrics
        self.fps_value: float = 0.0
        self.vram_usage: float = 0.0

        # Create UI components
        self._create_menu_bar()
        self._create_main_layout()
        self._create_status_bar()

        # Start update loop
        self._start_update_loop()

        # Configure window closing
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_menu_bar(self) -> None:
        """Create the top menu bar with File, View, Help menus."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Open Video", command=self._select_video_file, accelerator="Ctrl+O"
        )
        file_menu.add_command(
            label="Open Folder", command=self._select_video_folder, accelerator="Ctrl+Shift+O"
        )
        file_menu.add_separator()
        file_menu.add_command(label="Settings", command=self._open_settings, accelerator="Ctrl+,")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing, accelerator="Ctrl+Q")

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Agent Dashboard", command=self._toggle_agent_dashboard)
        view_menu.add_command(label="Toggle Theme", command=self._toggle_theme)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._open_docs)
        help_menu.add_command(label="About", command=self._show_about)

        # Keyboard shortcuts
        self.bind("<Control-o>", lambda e: self._select_video_file())
        self.bind("<Control-O>", lambda e: self._select_video_folder())
        self.bind("<Control-comma>", lambda e: self._open_settings())
        self.bind("<Control-q>", lambda e: self._on_closing())
        self.bind("<space>", lambda e: self._toggle_processing())

    def _create_main_layout(self) -> None:
        """Create the main application layout with three panels."""
        # Main container with padding
        main_container = ttk.Frame(self, padding="10")
        main_container.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Configure grid weights
        main_container.grid_rowconfigure(0, weight=3)  # Top section (video + agent dashboard)
        main_container.grid_rowconfigure(1, weight=0)  # Control panel
        main_container.grid_columnconfigure(0, weight=2)  # Left column (video preview)
        main_container.grid_columnconfigure(1, weight=1)  # Right column (agent dashboard)

        # Left panel: Video preview and controls
        self._create_video_panel(main_container)

        # Right panel: Agent dashboard
        self._create_agent_dashboard_panel(main_container)

        # Bottom panel: Control panel
        self._create_control_panel(main_container)

    def _create_video_panel(self, parent: ttk.Frame) -> None:
        """Create the video preview panel with file selection."""
        video_frame = ttk.LabelFrame(parent, text="Video Preview", padding="10")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        video_frame.grid_rowconfigure(1, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        # Video selection toolbar
        toolbar = ttk.Frame(video_frame)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ttk.Button(toolbar, text="Select Video", command=self._select_video_file).pack(
            side="left", padx=5
        )
        ttk.Button(toolbar, text="Select Folder", command=self._select_video_folder).pack(
            side="left", padx=5
        )

        self.video_path_label = ttk.Label(toolbar, text="No video selected", foreground="gray")
        self.video_path_label.pack(side="left", padx=10, fill="x", expand=True)

        # Video preview area (placeholder canvas)
        preview_container = ttk.Frame(video_frame, relief="sunken", borderwidth=2)
        preview_container.grid(row=1, column=0, sticky="nsew")
        preview_container.grid_rowconfigure(0, weight=1)
        preview_container.grid_columnconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(preview_container, bg="#2b2b2b", highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        # Placeholder text
        self.video_canvas.create_text(
            400,
            300,
            text="Video preview will appear here\n\nSelect a video file to begin",
            fill="gray",
            font=("Arial", 14),
            justify="center",
            tags="placeholder",
        )

        # Video info panel
        info_frame = ttk.Frame(video_frame)
        info_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        self.video_info_label = ttk.Label(
            info_frame, text="Resolution: N/A | FPS: N/A | Duration: N/A", foreground="gray"
        )
        self.video_info_label.pack(side="left")

    def _create_agent_dashboard_panel(self, parent: ttk.Frame) -> None:
        """Create the agent progress dashboard panel (unique feature)."""
        dashboard_frame = ttk.LabelFrame(parent, text="Agent Dashboard", padding="10")
        dashboard_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        dashboard_frame.grid_rowconfigure(0, weight=1)
        dashboard_frame.grid_columnconfigure(0, weight=1)

        # Create scrollable canvas for agent list
        canvas_container = ttk.Frame(dashboard_frame)
        canvas_container.grid(row=0, column=0, sticky="nsew")
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        self.agent_canvas = tk.Canvas(canvas_container, bg="#1e1e1e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            canvas_container, orient="vertical", command=self.agent_canvas.yview
        )

        self.agent_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.agent_canvas.configure(yscrollcommand=scrollbar.set)

        # Frame inside canvas for agent widgets
        self.agent_container = ttk.Frame(self.agent_canvas)
        self.agent_canvas_window = self.agent_canvas.create_window(
            (0, 0), window=self.agent_container, anchor="nw"
        )

        # Dictionary to store agent widgets
        self.agent_widgets: Dict[str, Dict[str, Any]] = {}

        # Create widgets for 15 agents
        self._create_agent_widgets()

        # Update scroll region
        self.agent_container.bind(
            "<Configure>",
            lambda e: self.agent_canvas.configure(scrollregion=self.agent_canvas.bbox("all")),
        )

        # Refresh button
        refresh_btn = ttk.Button(
            dashboard_frame, text="Refresh Agents", command=self._refresh_agent_status
        )
        refresh_btn.grid(row=1, column=0, pady=(10, 0))

    def _create_agent_widgets(self) -> None:
        """Create progress bar widgets for all 15 agents."""
        agents = [
            "project-architect",
            "requirements-analyst",
            "video-specialist",
            "ml-specialist",
            "tracker-dev-1",
            "tracker-dev-2",
            "ui-architect",
            "ui-enhancer",
            "cross-platform-dev",
            "test-engineer-1",
            "test-engineer-2",
            "integration-master",
            "code-quality",
            "gpu-debugger",
            "python-debugger",
        ]

        for idx, agent_name in enumerate(agents):
            agent_frame = ttk.Frame(self.agent_container)
            agent_frame.grid(row=idx, column=0, sticky="ew", pady=5, padx=10)
            agent_frame.grid_columnconfigure(0, weight=1)

            # Agent name label
            name_label = ttk.Label(agent_frame, text=agent_name, font=("Arial", 10, "bold"))
            name_label.grid(row=0, column=0, sticky="w")

            # Status label
            status_label = ttk.Label(
                agent_frame, text="PENDING", foreground="gray", font=("Arial", 9)
            )
            status_label.grid(row=0, column=1, sticky="e", padx=(10, 0))

            # Progress bar
            progress_bar = ttk.Progressbar(agent_frame, mode="determinate", length=300)
            progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
            progress_bar["value"] = 0

            # Store widgets for updates
            self.agent_widgets[agent_name] = {
                "frame": agent_frame,
                "name_label": name_label,
                "status_label": status_label,
                "progress_bar": progress_bar,
            }

    def _create_control_panel(self, parent: ttk.Frame) -> None:
        """Create the control panel with tracker selection and action buttons."""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        control_frame.grid_columnconfigure(1, weight=1)

        # Tracker selection
        ttk.Label(control_frame, text="Tracker:").grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.tracker_var = tk.StringVar(value="hybrid")
        tracker_combo = ttk.Combobox(
            control_frame,
            textvariable=self.tracker_var,
            values=["bytetrack", "botsort", "hybrid"],
            state="readonly",
            width=20,
        )
        tracker_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tracker_combo.bind("<<ComboboxSelected>>", self._on_tracker_changed)

        # Batch size control
        ttk.Label(control_frame, text="Batch Size:").grid(
            row=0, column=2, padx=5, pady=5, sticky="w"
        )

        self.batch_size_var = tk.IntVar(value=8)
        batch_spin = ttk.Spinbox(
            control_frame, from_=1, to=32, textvariable=self.batch_size_var, width=10
        )
        batch_spin.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Hardware acceleration checkbox
        self.hw_accel_var = tk.BooleanVar(value=True)
        hw_accel_check = ttk.Checkbutton(
            control_frame, text="Hardware Acceleration", variable=self.hw_accel_var
        )
        hw_accel_check.grid(row=0, column=4, padx=20, pady=5, sticky="w")

        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=5, padx=5, pady=5, sticky="e")

        self.start_btn = ttk.Button(
            button_frame, text="Start Processing", command=self._start_processing, width=15
        )
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(
            button_frame, text="Stop", command=self._stop_processing, state="disabled", width=10
        )
        self.stop_btn.pack(side="left", padx=5)

        # Performance metrics
        metrics_frame = ttk.Frame(control_frame)
        metrics_frame.grid(row=1, column=0, columnspan=6, pady=(10, 0), sticky="ew")

        ttk.Label(metrics_frame, text="Performance:").pack(side="left", padx=(0, 10))

        self.fps_label = ttk.Label(
            metrics_frame, text="FPS: 0.0", font=("Arial", 10, "bold"), foreground="green"
        )
        self.fps_label.pack(side="left", padx=10)

        self.vram_label = ttk.Label(
            metrics_frame, text="VRAM: 0.0 GB", font=("Arial", 10, "bold"), foreground="blue"
        )
        self.vram_label.pack(side="left", padx=10)

        self.frame_label = ttk.Label(metrics_frame, text="Frame: 0/0", foreground="gray")
        self.frame_label.pack(side="left", padx=10)

    def _create_status_bar(self) -> None:
        """Create the bottom status bar."""
        status_frame = ttk.Frame(self, relief="sunken", borderwidth=1)
        status_frame.grid(row=1, column=0, sticky="ew")

        self.status_label = ttk.Label(status_frame, text="Ready", padding=(10, 5))
        self.status_label.pack(side="left", fill="x", expand=True)

    # Event handlers

    def _select_video_file(self) -> None:
        """Open file dialog to select a video file."""
        filetypes = [("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(title="Select Video File", filetypes=filetypes)

        if filepath:
            self.video_path = Path(filepath)
            self.video_path_label.config(text=f"Selected: {self.video_path.name}")
            self.status_label.config(text=f"Video loaded: {self.video_path.name}")
            self._load_video_info()

    def _select_video_folder(self) -> None:
        """Open folder dialog to select a folder of videos."""
        folderpath = filedialog.askdirectory(title="Select Video Folder")

        if folderpath:
            self.video_path = Path(folderpath)
            video_files = list(self.video_path.glob("*.mp4")) + list(self.video_path.glob("*.avi"))
            self.video_path_label.config(
                text=f"Folder: {self.video_path.name} ({len(video_files)} videos)"
            )
            self.status_label.config(text=f"Folder loaded: {len(video_files)} videos found")

    def _load_video_info(self) -> None:
        """Load and display video metadata."""
        if self.video_path and self.video_path.is_file():
            # Placeholder - would integrate with VideoPipeline
            self.video_info_label.config(
                text="Resolution: 1920x1080 | FPS: 30.0 | Duration: 00:10:30"
            )

    def _toggle_processing(self) -> None:
        """Toggle between start and stop processing."""
        if self.is_processing:
            self._stop_processing()
        else:
            self._start_processing()

    def _start_processing(self) -> None:
        """Start video processing in a background thread."""
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video file or folder first.")
            return

        self.is_processing = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="Processing started...")

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_video, daemon=True)
        self.processing_thread.start()

    def _stop_processing(self) -> None:
        """Stop video processing."""
        self.is_processing = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Processing stopped.")

    def _process_video(self) -> None:
        """Background thread for video processing (placeholder)."""
        # This would integrate with the actual video pipeline
        frame_count = 0
        while self.is_processing and frame_count < 1000:
            frame_count += 1

            # Simulate processing
            time.sleep(0.033)  # ~30 FPS

            # Update metrics
            self.fps_value = 30.0 + (frame_count % 10)
            self.vram_usage = 5.2 + (frame_count % 100) / 100

            # Queue UI update
            self.update_queue.put(
                {
                    "fps": self.fps_value,
                    "vram": self.vram_usage,
                    "frame": frame_count,
                    "total": 1000,
                }
            )

        self.is_processing = False
        self.update_queue.put({"done": True})

    def _on_tracker_changed(self, event: Any) -> None:
        """Handle tracker selection change."""
        self.tracker_type = self.tracker_var.get()
        self.status_label.config(text=f"Tracker changed to: {self.tracker_type}")

    def _toggle_agent_dashboard(self) -> None:
        """Toggle agent dashboard visibility."""
        # Toggle visibility of agent dashboard
        pass

    def _toggle_theme(self) -> None:
        """Toggle between light and dark themes."""
        if SV_TTK_AVAILABLE:
            current = sv_ttk.get_theme()
            new_theme = "light" if current == "dark" else "dark"
            sv_ttk.set_theme(new_theme)
            self.status_label.config(text=f"Theme changed to: {new_theme}")

    def _open_settings(self) -> None:
        """Open settings panel in a new window."""
        # Would open SettingsPanel
        messagebox.showinfo("Settings", "Settings panel will be implemented in settings_panel.py")

    def _open_docs(self) -> None:
        """Open documentation."""
        messagebox.showinfo("Documentation", "Documentation: /home/pi/elo_elo_320/docs/")

    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = (
            "FunGen Rewrite v1.0\n\n"
            "AI-Powered Funscript Generator\n"
            "Built with Python 3.11 + tkinter\n\n"
            "Target: 100+ FPS on RTX 3090\n"
            "Platform: Raspberry Pi (dev) + RTX 3090 (prod)\n\n"
            "Author: ui-architect agent\n"
            "Date: 2025-10-24"
        )
        messagebox.showinfo("About FunGen Rewrite", about_text)

    def _refresh_agent_status(self) -> None:
        """Refresh agent progress from JSON files."""
        progress_dir = Path("/home/pi/elo_elo_320/progress")

        if not progress_dir.exists():
            return

        for agent_name, widgets in self.agent_widgets.items():
            progress_file = progress_dir / f"{agent_name}.json"

            if progress_file.exists():
                try:
                    with open(progress_file, "r") as f:
                        data = json.load(f)

                    progress = data.get("progress", 0)
                    status = data.get("status", "pending").upper()

                    # Update progress bar
                    widgets["progress_bar"]["value"] = progress

                    # Update status label with color
                    status_colors = {
                        "PENDING": "gray",
                        "IN_PROGRESS": "orange",
                        "WORKING": "orange",
                        "COMPLETED": "green",
                        "ERROR": "red",
                    }
                    color = status_colors.get(status, "gray")
                    widgets["status_label"].config(text=f"{status} ({progress}%)", foreground=color)

                except (json.JSONDecodeError, KeyError) as e:
                    pass

        self.status_label.config(text="Agent status refreshed")

    def _start_update_loop(self) -> None:
        """Start the UI update loop (runs every 100ms)."""
        self._process_update_queue()
        self._refresh_agent_status()  # Auto-refresh agents every cycle
        self.after(2000, self._start_update_loop)  # Refresh every 2 seconds

    def _process_update_queue(self) -> None:
        """Process updates from the processing thread."""
        try:
            while not self.update_queue.empty():
                update = self.update_queue.get_nowait()

                if "done" in update:
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.status_label.config(text="Processing complete!")
                else:
                    # Update performance metrics
                    if "fps" in update:
                        self.fps_label.config(text=f"FPS: {update['fps']:.1f}")
                    if "vram" in update:
                        self.vram_label.config(text=f"VRAM: {update['vram']:.1f} GB")
                    if "frame" in update and "total" in update:
                        self.frame_label.config(text=f"Frame: {update['frame']}/{update['total']}")
        except queue.Empty:
            pass

    def _on_closing(self) -> None:
        """Handle window closing event."""
        if self.is_processing:
            if messagebox.askokcancel("Quit", "Processing is active. Stop and quit?"):
                self.is_processing = False
                if self.processing_thread:
                    self.processing_thread.join(timeout=2.0)
                self.destroy()
        else:
            self.destroy()


def main():
    """Main entry point for testing the UI."""
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
