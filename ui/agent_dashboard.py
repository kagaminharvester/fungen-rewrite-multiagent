"""
FunGen Rewrite - Agent Dashboard

Real-time visualization of 15 agent progress bars (unique feature).
Reads progress/*.json files every 2 seconds and displays:
- Agent status (pending/in_progress/completed)
- Progress percentage (0-100%)
- Error/warning indicators
- Expandable logs
- Current task description

Author: ui-architect agent
Date: 2025-10-24
Platform: Cross-platform (Pi + RTX 3090)
"""

import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import scrolledtext, ttk
from typing import Any, Dict, List, Optional


class AgentDashboard(ttk.Frame):
    """
    Real-time agent progress visualization widget.

    Features:
        - Displays all 15 agents with progress bars
        - Color-coded status indicators
        - Auto-refresh every 2 seconds
        - Expandable log viewer
        - Click to view detailed agent info
        - Error/warning badges

    Attributes:
        progress_dir (Path): Directory containing agent progress JSON files
        agent_widgets (Dict): Dictionary of agent UI widgets
        auto_refresh (bool): Whether to auto-refresh agent status
        refresh_interval (int): Refresh interval in milliseconds
    """

    def __init__(self, parent: tk.Widget, progress_dir: str = "/home/pi/elo_elo_320/progress"):
        """
        Initialize the agent dashboard.

        Args:
            parent: Parent tkinter widget
            progress_dir: Directory containing agent progress JSON files
        """
        super().__init__(parent)

        self.progress_dir = Path(progress_dir)
        self.agent_widgets: Dict[str, Dict[str, Any]] = {}
        self.auto_refresh = True
        self.refresh_interval = 2000  # 2 seconds
        self.selected_agent: Optional[str] = None

        # Agent list (all 15 agents)
        self.agents = [
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

        self._create_widgets()
        self._start_auto_refresh()

    def _create_widgets(self) -> None:
        """Create the dashboard UI layout."""
        # Configure grid
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=3)  # Agent list
        self.grid_rowconfigure(2, weight=0)  # Controls
        self.grid_rowconfigure(3, weight=1)  # Detail view
        self.grid_columnconfigure(0, weight=1)

        # Header
        header_frame = ttk.Frame(self)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(header_frame, text="Agent Dashboard", font=("Arial", 14, "bold")).pack(
            side="left"
        )

        self.last_update_label = ttk.Label(
            header_frame, text="Last update: Never", foreground="gray", font=("Arial", 9)
        )
        self.last_update_label.pack(side="right")

        # Scrollable agent list
        self._create_agent_list()

        # Control buttons
        self._create_controls()

        # Detail view (expandable)
        self._create_detail_view()

    def _create_agent_list(self) -> None:
        """Create scrollable list of agent progress bars."""
        list_frame = ttk.LabelFrame(self, text="Agents (15 total)", padding="10")
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        # Create canvas with scrollbar
        canvas = tk.Canvas(list_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Container for agent widgets
        container = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=container, anchor="nw")

        # Create widgets for each agent
        for idx, agent_name in enumerate(self.agents):
            self._create_agent_widget(container, agent_name, idx)

        # Update scroll region
        container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Bind mousewheel for scrolling
        canvas.bind_all(
            "<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units")
        )

    def _create_agent_widget(self, parent: ttk.Frame, agent_name: str, row: int) -> None:
        """
        Create widget for a single agent.

        Args:
            parent: Parent frame
            agent_name: Name of the agent
            row: Grid row position
        """
        # Main frame for this agent
        agent_frame = ttk.Frame(parent, relief="solid", borderwidth=1, padding="5")
        agent_frame.grid(row=row, column=0, sticky="ew", pady=2, padx=5)
        agent_frame.grid_columnconfigure(0, weight=1)

        # Make clickable
        agent_frame.bind("<Button-1>", lambda e: self._on_agent_clicked(agent_name))

        # Top row: name and status
        top_row = ttk.Frame(agent_frame)
        top_row.grid(row=0, column=0, sticky="ew")
        top_row.grid_columnconfigure(0, weight=1)

        name_label = ttk.Label(top_row, text=agent_name, font=("Arial", 9, "bold"), cursor="hand2")
        name_label.grid(row=0, column=0, sticky="w")
        name_label.bind("<Button-1>", lambda e: self._on_agent_clicked(agent_name))

        status_label = ttk.Label(top_row, text="PENDING", foreground="gray", font=("Arial", 8))
        status_label.grid(row=0, column=1, sticky="e", padx=(5, 0))

        # Progress bar
        progress_bar = ttk.Progressbar(agent_frame, mode="determinate", length=200)
        progress_bar.grid(row=1, column=0, sticky="ew", pady=(3, 0))
        progress_bar["value"] = 0

        # Percentage label
        percent_label = ttk.Label(agent_frame, text="0%", font=("Arial", 8), foreground="gray")
        percent_label.grid(row=1, column=1, sticky="e", padx=(5, 0))

        # Task description (hidden by default)
        task_label = ttk.Label(
            agent_frame, text="", font=("Arial", 8, "italic"), foreground="gray", wraplength=350
        )
        task_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(3, 0))

        # Store widgets
        self.agent_widgets[agent_name] = {
            "frame": agent_frame,
            "name_label": name_label,
            "status_label": status_label,
            "progress_bar": progress_bar,
            "percent_label": percent_label,
            "task_label": task_label,
        }

    def _create_controls(self) -> None:
        """Create control buttons."""
        control_frame = ttk.Frame(self)
        control_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ttk.Button(control_frame, text="Refresh Now", command=self.refresh_agents).pack(
            side="left", padx=5
        )

        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_frame,
            text="Auto-refresh (2s)",
            variable=self.auto_refresh_var,
            command=self._toggle_auto_refresh,
        ).pack(side="left", padx=5)

        ttk.Button(control_frame, text="Clear Selection", command=self._clear_selection).pack(
            side="left", padx=5
        )

        # Summary stats
        self.summary_label = ttk.Label(
            control_frame,
            text="Completed: 0/15 | In Progress: 0 | Pending: 15",
            foreground="gray",
            font=("Arial", 9),
        )
        self.summary_label.pack(side="right", padx=5)

    def _create_detail_view(self) -> None:
        """Create expandable detail view for selected agent."""
        detail_frame = ttk.LabelFrame(self, text="Agent Details", padding="10")
        detail_frame.grid(row=3, column=0, sticky="nsew")
        detail_frame.grid_rowconfigure(0, weight=1)
        detail_frame.grid_columnconfigure(0, weight=1)

        # Scrolled text for JSON details
        self.detail_text = scrolledtext.ScrolledText(
            detail_frame, wrap=tk.WORD, height=10, font=("Courier", 9)
        )
        self.detail_text.grid(row=0, column=0, sticky="nsew")
        self.detail_text.insert("1.0", "Click on an agent to view details...")
        self.detail_text.config(state="disabled")

    def refresh_agents(self) -> None:
        """Refresh agent status from JSON files."""
        if not self.progress_dir.exists():
            return

        stats = {"completed": 0, "in_progress": 0, "pending": 0, "total": 0}

        for agent_name, widgets in self.agent_widgets.items():
            progress_file = self.progress_dir / f"{agent_name}.json"

            if progress_file.exists():
                try:
                    with open(progress_file, "r") as f:
                        data = json.load(f)

                    self._update_agent_widget(agent_name, data, stats)

                except (json.JSONDecodeError, KeyError, IOError) as e:
                    # Show error status
                    widgets["status_label"].config(text="ERROR", foreground="red")
                    widgets["percent_label"].config(text="N/A")
            else:
                # No progress file yet
                widgets["status_label"].config(text="PENDING", foreground="gray")
                widgets["progress_bar"]["value"] = 0
                widgets["percent_label"].config(text="0%")
                stats["pending"] += 1

            stats["total"] += 1

        # Update summary
        self.summary_label.config(
            text=f"Completed: {stats['completed']}/{stats['total']} | "
            f"In Progress: {stats['in_progress']} | "
            f"Pending: {stats['pending']}"
        )

        # Update timestamp
        self.last_update_label.config(text=f"Last update: {datetime.now().strftime('%H:%M:%S')}")

    def _update_agent_widget(
        self, agent_name: str, data: Dict[str, Any], stats: Dict[str, int]
    ) -> None:
        """
        Update a single agent widget with new data.

        Args:
            agent_name: Name of the agent
            data: Progress data from JSON
            stats: Statistics dictionary to update
        """
        widgets = self.agent_widgets[agent_name]

        progress = data.get("progress", 0)
        status = data.get("status", "pending").upper()
        current_task = data.get("current_task", "")

        # Update progress bar
        widgets["progress_bar"]["value"] = progress
        widgets["percent_label"].config(text=f"{progress}%")

        # Update status with color
        status_colors = {
            "PENDING": ("gray", "pending"),
            "IN_PROGRESS": ("orange", "in_progress"),
            "WORKING": ("orange", "in_progress"),
            "COMPLETED": ("green", "completed"),
            "DONE": ("green", "completed"),
            "ERROR": ("red", "pending"),
        }

        color, stat_key = status_colors.get(status, ("gray", "pending"))
        widgets["status_label"].config(text=status, foreground=color)

        # Update stats
        stats[stat_key] = stats.get(stat_key, 0) + 1

        # Update task description
        if current_task:
            widgets["task_label"].config(text=f"Task: {current_task}")
        else:
            widgets["task_label"].config(text="")

    def _on_agent_clicked(self, agent_name: str) -> None:
        """
        Handle agent widget click.

        Args:
            agent_name: Name of the clicked agent
        """
        self.selected_agent = agent_name

        # Highlight selected agent
        for name, widgets in self.agent_widgets.items():
            if name == agent_name:
                widgets["frame"].config(relief="raised", borderwidth=2)
            else:
                widgets["frame"].config(relief="solid", borderwidth=1)

        # Load and display details
        self._load_agent_details(agent_name)

    def _load_agent_details(self, agent_name: str) -> None:
        """
        Load and display detailed agent info.

        Args:
            agent_name: Name of the agent
        """
        progress_file = self.progress_dir / f"{agent_name}.json"

        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", tk.END)

        if progress_file.exists():
            try:
                with open(progress_file, "r") as f:
                    data = json.load(f)

                # Format JSON nicely
                formatted = json.dumps(data, indent=2)
                self.detail_text.insert("1.0", formatted)
            except (json.JSONDecodeError, IOError) as e:
                self.detail_text.insert("1.0", f"Error loading agent details:\n{str(e)}")
        else:
            self.detail_text.insert(
                "1.0",
                f"No progress file found for {agent_name}\n\n"
                f"Expected location: {progress_file}",
            )

        self.detail_text.config(state="disabled")

    def _clear_selection(self) -> None:
        """Clear agent selection."""
        self.selected_agent = None

        for widgets in self.agent_widgets.values():
            widgets["frame"].config(relief="solid", borderwidth=1)

        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert("1.0", "Click on an agent to view details...")
        self.detail_text.config(state="disabled")

    def _toggle_auto_refresh(self) -> None:
        """Toggle auto-refresh on/off."""
        self.auto_refresh = self.auto_refresh_var.get()

    def _start_auto_refresh(self) -> None:
        """Start auto-refresh loop."""
        if self.auto_refresh:
            self.refresh_agents()

        self.after(self.refresh_interval, self._start_auto_refresh)


def main():
    """Test the AgentDashboard widget."""
    root = tk.Tk()
    root.title("Agent Dashboard Test")
    root.geometry("600x800")

    dashboard = AgentDashboard(root)
    dashboard.pack(fill="both", expand=True, padx=10, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
