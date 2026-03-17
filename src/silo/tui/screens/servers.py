"""Servers screen — process management with log tailing across nodes."""

from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static
from textual import work

from silo.config.paths import LOGS_DIR
from silo.tui.widgets.confirm_modal import ConfirmModal
from silo.tui.widgets.log_viewer import LogViewer
from silo.tui.widgets.nav_bar import NavBar
from silo.tui.widgets.status_counts import StatusCounts


class ServersScreen(Screen):
    """Server process management: up, down, logs — across all nodes."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("enter", "toggle", "Toggle Up/Down"),
        ("u", "up_selected", "Up Selected"),
        ("d", "down_selected", "Down Selected"),
        ("U", "up_all", "Up All"),
        ("D", "down_all", "Down All"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatusCounts(id="status-counts")
        yield Static(" [b]Model Servers[/b]", classes="section-title")
        yield DataTable(id="server-table")
        yield Static(" [b]Logs[/b]  [dim](local only)[/]", classes="section-title")
        yield LogViewer(id="log-viewer")
        yield NavBar(active_screen="servers")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#server-table", DataTable)
        table.add_columns("NODE", "NAME", "REPO", "PORT", "PID", "STATUS")
        table.cursor_type = "row"
        self.action_refresh()
        self.set_interval(5.0, self._poll_status)

    def action_refresh(self) -> None:
        self._load_servers()

    def _build_clients(self):
        from silo.agent.client import build_clients
        from silo.config.loader import load_config

        config = load_config()
        return build_clients(config.nodes), config

    @work(thread=True)
    def _load_servers(self) -> None:
        clients, config = self._build_clients()

        # Row: (node, name, repo, port, pid, status)
        rows: list[tuple[str, str, str, str, str, str]] = []
        total_registered = 0

        for node_name, client in clients.items():
            try:
                # Get processes from this node
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
                    pid_str = str(info.pid) if info.pid else "—"
                    rows.append((
                        node_name,
                        model_cfg.name,
                        model_cfg.repo,
                        str(model_cfg.port),
                        pid_str,
                        info.status,
                    ))

                # Also show PID-only processes on this node
                for proc in client.list_processes():
                    if proc.name not in config_names:
                        rows.append((
                            node_name,
                            proc.name,
                            proc.repo_id,
                            str(proc.port),
                            str(proc.pid),
                            proc.status,
                        ))

                total_registered += len(client.registry())

            except Exception:
                # Node unreachable — show it as offline
                for model_cfg in config.models:
                    if (model_cfg.node or "local") == node_name:
                        rows.append((
                            node_name,
                            model_cfg.name,
                            model_cfg.repo,
                            str(model_cfg.port),
                            "—",
                            "unreachable",
                        ))

        # Get local memory for the status bar
        try:
            local_client = clients["local"]
            mem = local_client.memory()
            mem_pct = mem.usage_percent
            mem_pressure = mem.pressure
        except Exception:
            mem_pct = 0.0
            mem_pressure = "unknown"

        running = sum(1 for r in rows if r[5] == "running")
        self.app.call_from_thread(
            self._apply_servers,
            rows,
            running,
            total_registered,
            mem_pct,
            mem_pressure,
        )

    def _apply_servers(self, rows, running, registered, mem_pct, mem_pressure) -> None:
        counts = self.query_one(StatusCounts)
        counts.running = running
        counts.registered = registered
        counts.memory_pct = mem_pct
        counts.memory_pressure = mem_pressure

        table = self.query_one("#server-table", DataTable)
        # Preserve cursor position across refresh
        prev_cursor = table.cursor_row
        table.clear()
        for node, name, repo, port, pid, status in rows:
            style = {
                "running": "green",
                "unreachable": "red",
            }.get(status, "dim")
            table.add_row(
                f"[{style}]{node}[/]",
                f"[{style}]{name}[/]",
                f"[{style}]{repo}[/]",
                f"[{style}]{port}[/]",
                f"[{style}]{pid}[/]",
                f"[{style}]{status}[/]",
            )
        # Restore cursor position
        if prev_cursor is not None and rows:
            table.move_cursor(row=min(prev_cursor, len(rows) - 1))

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        if event.row_key is None:
            return
        table = self.query_one("#server-table", DataTable)
        row_idx = list(table.rows.keys()).index(event.row_key)
        node = self._strip_markup(str(table.get_cell_at((row_idx, 0))))
        name = self._strip_markup(str(table.get_cell_at((row_idx, 1))))
        # Only tail logs for local node
        if node == "local":
            log_path = LOGS_DIR / f"{name}.log"
            viewer = self.query_one(LogViewer)
            viewer.log_path = log_path

    @staticmethod
    def _strip_markup(text: str) -> str:
        return re.sub(r"\[/?[^\]]*\]", "", text)

    def _get_selected(self) -> tuple[str, str] | None:
        """Return (node_name, model_name) of selected row, or None."""
        table = self.query_one("#server-table", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            return None
        node = self._strip_markup(str(table.get_cell_at((table.cursor_row, 0))))
        name = self._strip_markup(str(table.get_cell_at((table.cursor_row, 1))))
        return node, name

    def _get_model_config(self, name: str):
        from silo.config.loader import load_config

        config = load_config()
        for m in config.models:
            if m.name == name:
                return m
        return None

    def _get_client(self, node_name: str):
        clients, _ = self._build_clients()
        return clients.get(node_name)

    def action_toggle(self) -> None:
        selected = self._get_selected()
        if not selected:
            return
        node_name, name = selected
        model_cfg = self._get_model_config(name)
        client = self._get_client(node_name)
        if not client:
            return
        info = client.get_status(
            name,
            port=model_cfg.port if model_cfg else 0,
            repo_id=model_cfg.repo if model_cfg else "",
        )
        if info.status == "running":
            self._confirm_stop(node_name, name)
        elif model_cfg:
            self._do_start(node_name, name, model_cfg)
        else:
            self.notify(f"No config found for '{name}'", severity="warning")

    def action_up_selected(self) -> None:
        selected = self._get_selected()
        if not selected:
            return
        node_name, name = selected
        model_cfg = self._get_model_config(name)
        if model_cfg:
            self._do_start(node_name, name, model_cfg)
        else:
            self.notify(f"No config for '{name}'", severity="warning")

    def action_down_selected(self) -> None:
        selected = self._get_selected()
        if selected:
            self._confirm_stop(selected[0], selected[1])

    def _confirm_stop(self, node_name: str, name: str) -> None:
        """Show confirmation before stopping a model."""
        def on_confirm(confirmed: bool) -> None:
            if confirmed:
                self._do_stop(node_name, name)

        self.app.push_screen(
            ConfirmModal(f"Stop '{name}' on {node_name}?"),
            on_confirm,
        )

    def action_up_all(self) -> None:
        from silo.config.loader import load_config

        config = load_config()
        for model_cfg in config.models:
            node = model_cfg.node or "local"
            self._do_start(node, model_cfg.name, model_cfg)

    def action_down_all(self) -> None:
        def on_confirm(confirmed: bool) -> None:
            if not confirmed:
                return
            from silo.config.loader import load_config

            config = load_config()
            for model_cfg in config.models:
                node = model_cfg.node or "local"
                self._do_stop(node, model_cfg.name)

        self.app.push_screen(
            ConfirmModal("Stop ALL running servers?"),
            on_confirm,
        )

    @work(thread=True)
    def _do_start(self, node_name: str, name: str, model_cfg) -> None:
        client = self._get_client(node_name)
        if not client:
            self.app.call_from_thread(
                self.notify, f"Node '{node_name}' not found", severity="error"
            )
            return
        try:
            pid = client.spawn(
                name=model_cfg.name,
                repo_id=model_cfg.repo,
                host=model_cfg.host,
                port=model_cfg.port,
                quantize=model_cfg.quantize,
                output=model_cfg.output,
            )
            self.app.call_from_thread(
                self.notify, f"Started {name} on {node_name} (PID {pid})"
            )
        except Exception as exc:
            self.app.call_from_thread(
                self.notify, f"Failed to start {name}: {exc}", severity="error"
            )
        self.app.call_from_thread(self.action_refresh)

    @work(thread=True)
    def _do_stop(self, node_name: str, name: str) -> None:
        client = self._get_client(node_name)
        if not client:
            self.app.call_from_thread(
                self.notify, f"Node '{node_name}' not found", severity="error"
            )
            return
        try:
            stopped = client.stop(name)
            msg = f"Stopped {name} on {node_name}" if stopped else f"{name} not found"
            self.app.call_from_thread(self.notify, msg)
        except Exception as exc:
            self.app.call_from_thread(
                self.notify, f"Failed to stop {name}: {exc}", severity="error"
            )
        self.app.call_from_thread(self.action_refresh)

    def _poll_status(self) -> None:
        self._load_servers()
