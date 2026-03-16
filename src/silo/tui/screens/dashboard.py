"""Dashboard screen — fleet overview with nodes, servers, and downloads."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from silo.tui.widgets.nav_bar import NavBar
from silo.tui.widgets.status_counts import StatusCounts
from silo.tui.widgets.wake_status import WakeStatusBar


class DashboardScreen(Screen):
    """Fleet overview: nodes, running servers, active downloads."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatusCounts(id="status-counts")
        yield WakeStatusBar(id="wake-status")
        yield Static(" [b]Nodes[/b]", classes="section-title")
        yield DataTable(id="nodes-table")
        yield Static(" [b]Servers[/b]", classes="section-title")
        yield DataTable(id="servers-table")
        yield Static(" [b]Downloads[/b]", classes="section-title")
        yield DataTable(id="downloads-table")
        yield NavBar(active_screen="dashboard")
        yield Footer()

    def on_mount(self) -> None:
        nodes = self.query_one("#nodes-table", DataTable)
        nodes.add_columns("NODE", "STATUS", "MEMORY", "MODELS", "RUNNING")
        nodes.cursor_type = "row"

        servers = self.query_one("#servers-table", DataTable)
        servers.add_columns("NODE", "NAME", "PORT", "STATUS")
        servers.cursor_type = "row"

        downloads = self.query_one("#downloads-table", DataTable)
        downloads.add_columns("REPO", "PROGRESS", "SPEED", "ETA", "STATUS")
        downloads.cursor_type = "row"

        self.action_refresh()
        self._dl_timer = self.set_interval(2.0, self._refresh_downloads)
        self._wake_timer = self.set_interval(1.0, self._refresh_wake_status)

    def action_refresh(self) -> None:
        self._load_data()

    @work(thread=True)
    def _load_data(self) -> None:
        from silo.agent.client import build_clients
        from silo.config.loader import load_config

        config = load_config()
        clients = build_clients(config.nodes)

        # ── Nodes table data ──
        node_rows: list[tuple[str, str, str, str, str]] = []
        # ── Servers table data ──
        server_rows: list[tuple[str, str, str, str]] = []
        total_running = 0
        total_registered = 0
        seen_nodes: set[str] = set()

        for node_name, client in clients.items():
            seen_nodes.add(node_name)
            try:
                mem = client.memory()
                registry = client.registry()
                processes = client.list_processes()

                running_on_node = sum(
                    1 for p in processes if p.status == "running"
                )
                total_running += running_on_node
                total_registered += len(registry)

                mem_str = (
                    f"{mem.usage_percent:.0f}% of {mem.total_gb:.0f} GB"
                )
                pressure_color = {
                    "normal": "green",
                    "warn": "yellow",
                    "critical": "red",
                }.get(mem.pressure, "dim")

                node_rows.append((
                    node_name,
                    "[green]online[/]",
                    f"[{pressure_color}]{mem_str}[/]",
                    str(len(registry)),
                    str(running_on_node),
                ))

                # Server rows for this node
                node_models = [
                    m for m in config.models
                    if (m.node or "local") == node_name
                ]
                config_names = {m.name for m in node_models}

                for model_cfg in node_models:
                    info = client.get_status(
                        model_cfg.name,
                        port=model_cfg.port,
                        repo_id=model_cfg.repo,
                    )
                    server_rows.append((
                        node_name, model_cfg.name,
                        str(model_cfg.port), info.status,
                    ))

                for proc in processes:
                    if proc.name not in config_names:
                        server_rows.append((
                            node_name, proc.name,
                            str(proc.port), proc.status,
                        ))

            except Exception:
                node_rows.append((
                    node_name,
                    "[red]offline[/]",
                    "—",
                    "—",
                    "—",
                ))
                for model_cfg in config.models:
                    if (model_cfg.node or "local") == node_name:
                        server_rows.append((
                            node_name, model_cfg.name,
                            str(model_cfg.port), "unreachable",
                        ))

        # ── Merge cluster workers not already in config ──
        cluster_data = self._fetch_cluster_status()
        if cluster_data is not None:
            for w in cluster_data.get("workers", []):
                if w["name"] in seen_nodes:
                    continue
                seen_nodes.add(w["name"])
                status = w.get("status", "unknown")
                mem = w.get("memory")
                processes = w.get("processes", [])
                running_on_node = sum(
                    1 for p in processes if p.get("status") == "running"
                )
                total_running += running_on_node

                if mem:
                    mem_str = f"{mem['usage_percent']:.0f}% of {mem['total_gb']:.0f} GB"
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

                node_rows.append((
                    w["name"],
                    f"[{status_color}]{status}[/]",
                    mem_display,
                    "—",
                    str(running_on_node),
                ))

                for proc in processes:
                    proc_status = proc.get("status", "unknown")
                    server_rows.append((
                        w["name"],
                        proc["name"],
                        str(proc.get("port", "—")),
                        proc_status,
                    ))

        # Local memory for status bar
        try:
            local_mem = clients["local"].memory()
            mem_pct = local_mem.usage_percent
            mem_pressure = local_mem.pressure
        except Exception:
            mem_pct = 0.0
            mem_pressure = "unknown"

        self.app.call_from_thread(
            self._apply_data,
            node_rows,
            server_rows,
            total_running,
            total_registered,
            mem_pct,
            mem_pressure,
        )

    def _fetch_cluster_status(self) -> dict | None:
        """Fetch cluster status from the head node, if available."""
        import json
        import urllib.error
        import urllib.request

        head_port = getattr(self.app, "agent_head_port", None)
        if head_port is not None:
            head_url = f"http://127.0.0.1:{head_port}"
        else:
            head_url = getattr(self.app, "cluster_head_url", None)
        if head_url is None:
            return None
        try:
            url = f"{head_url}/cluster/status"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception:
            return None

    def _apply_data(
        self,
        node_rows,
        server_rows,
        running_count,
        registered_count,
        mem_pct,
        mem_pressure,
    ) -> None:
        counts = self.query_one(StatusCounts)
        counts.running = running_count
        counts.registered = registered_count
        counts.memory_pct = mem_pct
        counts.memory_pressure = mem_pressure

        # ── Nodes ──
        nodes_t = self.query_one("#nodes-table", DataTable)
        nodes_t.clear()
        for node, status, mem, models, running in node_rows:
            nodes_t.add_row(f"[bold]{node}[/]", status, mem, models, running)
        if not node_rows:
            nodes_t.add_row("[dim]—[/]", "—", "—", "—", "—")

        # ── Servers ──
        servers_t = self.query_one("#servers-table", DataTable)
        servers_t.clear()
        for node, name, port, status in server_rows:
            style = {
                "running": "green",
                "unreachable": "red",
            }.get(status, "dim")
            servers_t.add_row(
                f"[{style}]{node}[/]",
                f"[{style}]{name}[/]",
                f"[{style}]{port}[/]",
                f"[{style}]{status}[/]",
            )
        if not server_rows:
            servers_t.add_row(
                "[dim]—[/]", "[dim]—[/]", "[dim]—[/]",
                "[dim]No models configured[/]",
            )

        # Initial download refresh
        self._refresh_downloads()

    def _refresh_downloads(self) -> None:
        """Update the downloads table from the tracker."""
        tracker = getattr(self.app, "downloads", None)
        if tracker is None:
            return

        # Poll filesystem to update progress for active downloads
        tracker.poll_active_progress()

        table = self.query_one("#downloads-table", DataTable)
        table.clear()

        entries = tracker.recent(limit=10)
        if entries:
            for entry in entries:
                status_style = {
                    "downloading": "yellow",
                    "done": "green",
                    "failed": "red",
                    "pending": "dim",
                }.get(entry.status, "dim")

                status_text = str(entry.status)
                if entry.status == "downloading":
                    status_text = f"downloading ({entry.elapsed_str})"
                elif entry.status == "done":
                    status_text = f"done ({entry.elapsed_str})"

                table.add_row(
                    entry.repo_id,
                    f"[{status_style}]{entry.progress_str}[/]",
                    f"[{status_style}]{entry.speed_str}[/]",
                    f"[{status_style}]{entry.eta_str}[/]",
                    f"[{status_style}]{status_text}[/]",
                )
        else:
            table.add_row(
                "[dim]No downloads[/]", "—", "—", "—", "—"
            )

    def _refresh_wake_status(self) -> None:
        """Update the wake status bar from the app-level wake state."""
        wake_state = getattr(self.app, "wake_status", None)
        bar = self.query_one("#wake-status", WakeStatusBar)
        if wake_state is None:
            bar.state = "off"
            return
        bar.state = str(wake_state.state)
        bar.wake_word = wake_state.wake_word
        bar.flow_name = wake_state.flow_name
        bar.detections = wake_state.detections
        bar.error = wake_state.error or ""
