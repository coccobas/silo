"""Cluster screen — head node management with workers, spawn, and stop."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from silo.tui.widgets.nav_bar import NavBar
from silo.tui.widgets.status_counts import StatusCounts


class ClusterScreen(Screen):
    """Cluster management: workers, models across nodes, spawn/stop."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("g", "register", "Register"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatusCounts(id="status-counts")
        yield Static(" [b]Workers[/b]", classes="section-title")
        yield DataTable(id="workers-table")
        yield Static(" [b]Models (cluster-wide)[/b]", classes="section-title")
        yield DataTable(id="cluster-models-table")
        yield Static(
            " [dim]r[/] refresh  [dim]g[/] register worker",
            classes="hint-bar",
        )
        yield NavBar(active_screen="cluster")
        yield Footer()

    def on_mount(self) -> None:
        workers = self.query_one("#workers-table", DataTable)
        workers.add_columns("NAME", "HOST", "PORT", "STATUS", "MEMORY", "MODELS")
        workers.cursor_type = "row"

        models = self.query_one("#cluster-models-table", DataTable)
        models.add_columns("NODE", "NAME", "REPO", "PORT", "PID", "STATUS")
        models.cursor_type = "row"

        self.action_refresh()
        self._poll_timer = self.set_interval(5.0, self.action_refresh)

    def action_refresh(self) -> None:
        self._load_data()

    @work(thread=True)
    def _load_data(self) -> None:
        import json
        import urllib.error
        import urllib.request

        head_url = self._get_head_url()
        if head_url is None:
            self.app.call_from_thread(self._apply_no_head)
            return

        try:
            url = f"{head_url}/cluster/status"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except Exception:
            self.app.call_from_thread(self._apply_no_head)
            return

        worker_rows: list[tuple[str, str, str, str, str, str]] = []
        model_rows: list[tuple[str, str, str, str, str, str]] = []
        total_running = 0
        total_available = 0.0
        total_memory = 0.0

        for w in data.get("workers", []):
            status = w.get("status", "unknown")
            mem = w.get("memory")
            processes = w.get("processes", [])
            running_count = sum(
                1 for p in processes if p.get("status") == "running"
            )
            total_running += running_count

            if mem:
                mem_str = f"{mem['usage_percent']:.0f}% of {mem['total_gb']:.0f} GB"
                total_memory += mem["total_gb"]
                total_available += mem["available_gb"]
                pressure_color = {
                    "normal": "green",
                    "warn": "yellow",
                    "critical": "red",
                }.get(mem.get("pressure", ""), "dim")
                mem_display = f"[{pressure_color}]{mem_str}[/]"
            else:
                mem_display = "[dim]—[/]"

            status_color = {
                "healthy": "green",
                "unhealthy": "red",
                "unknown": "yellow",
            }.get(status, "dim")

            worker_rows.append((
                w["name"],
                w.get("host", "—"),
                str(w.get("port", "—")),
                f"[{status_color}]{status}[/]",
                mem_display,
                str(running_count),
            ))

            for proc in processes:
                proc_status = proc.get("status", "unknown")
                proc_color = "green" if proc_status == "running" else "dim"
                model_rows.append((
                    f"[{proc_color}]{w['name']}[/]",
                    f"[{proc_color}]{proc['name']}[/]",
                    f"[{proc_color}]{proc.get('repo_id', '—')}[/]",
                    f"[{proc_color}]{proc.get('port', '—')}[/]",
                    f"[{proc_color}]{proc.get('pid', '—')}[/]",
                    f"[{proc_color}]{proc_status}[/]",
                ))

        mem_pct = (
            ((total_memory - total_available) / total_memory * 100)
            if total_memory > 0
            else 0.0
        )

        self.app.call_from_thread(
            self._apply_data,
            worker_rows,
            model_rows,
            total_running,
            len(data.get("workers", [])),
            mem_pct,
        )

    def _apply_data(
        self,
        worker_rows: list[tuple[str, str, str, str, str, str]],
        model_rows: list[tuple[str, str, str, str, str, str]],
        running_count: int,
        worker_count: int,
        mem_pct: float,
    ) -> None:
        counts = self.query_one(StatusCounts)
        counts.running = running_count
        counts.registered = worker_count
        counts.memory_pct = mem_pct

        workers_t = self.query_one("#workers-table", DataTable)
        workers_t.clear()
        for name, host, port, status, mem, models in worker_rows:
            workers_t.add_row(f"[bold]{name}[/]", host, port, status, mem, models)
        if not worker_rows:
            workers_t.add_row(
                "[dim]—[/]", "—", "—", "—", "—",
                "[dim]No workers registered[/]",
            )

        models_t = self.query_one("#cluster-models-table", DataTable)
        models_t.clear()
        for row in model_rows:
            models_t.add_row(*row)
        if not model_rows:
            models_t.add_row(
                "[dim]—[/]", "[dim]—[/]", "[dim]—[/]",
                "[dim]—[/]", "[dim]—[/]",
                "[dim]No models running[/]",
            )

    def _apply_no_head(self) -> None:
        """Show empty state when no head node is reachable."""
        counts = self.query_one(StatusCounts)
        counts.running = 0
        counts.registered = 0
        counts.memory_pct = 0.0
        counts.memory_pressure = "unknown"

        workers_t = self.query_one("#workers-table", DataTable)
        workers_t.clear()
        workers_t.add_row(
            "[red]No head node[/]", "—", "—", "—", "—",
            "[dim]Start a node with --head[/]",
        )

        models_t = self.query_one("#cluster-models-table", DataTable)
        models_t.clear()

    def action_register(self) -> None:
        """Show a prompt to register a new worker."""
        from silo.tui.widgets.register_modal import RegisterModal

        def on_result(result: dict | None) -> None:
            if result is not None:
                self._do_register(result)

        self.app.push_screen(RegisterModal(), on_result)

    @work(thread=True)
    def _do_register(self, data: dict) -> None:
        import json
        import urllib.request

        head_url = self._get_head_url()
        if head_url is None:
            return

        try:
            body = json.dumps(data).encode()
            req = urllib.request.Request(
                f"{head_url}/cluster/register",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception:
            pass

        # Refresh after registration
        self._load_data()

    def _get_head_url(self) -> str | None:
        """Find the head node URL from config or app state."""
        try:
            # If the TUI was launched with --head, use that port
            head_port = getattr(self.app, "agent_head_port", None)
            if head_port is not None:
                return f"http://127.0.0.1:{head_port}"

            from silo.config.loader import load_config

            config = load_config()
            # Check for a node marked as head, or use first node, or localhost
            for node in config.nodes:
                return f"http://{node.host}:{node.port}"
            # Default: assume head is running locally
            return "http://127.0.0.1:9900"
        except Exception:
            return None
