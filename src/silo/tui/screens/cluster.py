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
        ("e", "edit_model", "Edit"),
        ("g", "register", "Register"),
        ("s", "spawn", "Spawn"),
        ("x", "stop_model", "Stop"),
        ("d", "download", "Download"),
        ("delete", "remove_worker", "Remove"),
    ]

    # Cache worker names for modals
    _worker_names: list[str] = []
    # Cache model entries: (name, worker_host, model_port)
    _model_entries: list[tuple[str, str, int]] = []
    # Cache model names for stop (derived from _model_entries)
    _model_names: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatusCounts(id="status-counts")
        yield Static(" [b]Workers[/b]", classes="section-title")
        yield DataTable(id="workers-table")
        yield Static(" [b]Models (cluster-wide)[/b]", classes="section-title")
        yield DataTable(id="cluster-models-table")
        yield Static(
            " [dim]r[/] refresh  [dim]e[/] edit  [dim]g[/] register  "
            "[dim]s[/] spawn  [dim]x[/] stop  [dim]d[/] download  "
            "[dim]del[/] remove",
            classes="hint-bar",
        )
        yield NavBar(active_screen="cluster")
        yield Footer()

    def on_mount(self) -> None:
        workers = self.query_one("#workers-table", DataTable)
        workers.add_columns(
            "NAME", "HOST", "PORT", "STATUS", "VERSION",
            "CPU", "GPU", "MEMORY", "MODELS",
        )
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
            self.app.call_from_thread(
                self._apply_no_head, "Searching for head node..."
            )
            return

        try:
            url = f"{head_url}/cluster/status"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            self.app.call_from_thread(
                self._apply_no_head,
                f"Cannot reach head at {head_url}: {type(exc).__name__}",
            )
            return

        from silo import __version__

        worker_rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
        model_rows: list[tuple[str, str, str, str, str, str]] = []
        worker_names: list[str] = []
        model_names: list[str] = []
        model_entries: list[tuple[str, str, int]] = []
        total_running = 0
        total_available = 0.0
        total_memory = 0.0

        for w in data.get("workers", []):
            worker_names.append(w["name"])
            status = w.get("status", "unknown")
            mem = w.get("memory")
            processes = w.get("processes", [])
            version = w.get("version")
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

            # Version display with mismatch warning
            if version is None:
                version_display = "[dim]—[/]"
            elif version != __version__:
                version_display = f"[red]{version}[/]"
            else:
                version_display = f"[green]{version}[/]"

            sys_stats = w.get("system_stats")
            if sys_stats:
                cpu_display = f"{sys_stats['cpu_percent']:.0f}%"
                gpu_display = (
                    f"{sys_stats['gpu_percent']:.0f}%"
                    if sys_stats["gpu_percent"] > 0
                    else "[dim]—[/]"
                )
            else:
                cpu_display = "[dim]—[/]"
                gpu_display = "[dim]—[/]"

            worker_rows.append((
                w["name"],
                w.get("host", "—"),
                str(w.get("port", "—")),
                f"[{status_color}]{status}[/]",
                version_display,
                cpu_display,
                gpu_display,
                mem_display,
                str(running_count),
            ))

            for proc in processes:
                proc_status = proc.get("status", "unknown")
                proc_color = "green" if proc_status == "running" else "dim"
                model_names.append(proc["name"])
                model_entries.append((
                    proc["name"],
                    w.get("host", ""),
                    int(proc.get("port", 0)),
                ))
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
            worker_names,
            model_names,
            model_entries,
        )

    def _apply_data(
        self,
        worker_rows: list[tuple[str, str, str, str, str, str, str, str, str]],
        model_rows: list[tuple[str, str, str, str, str, str]],
        running_count: int,
        worker_count: int,
        mem_pct: float,
        worker_names: list[str],
        model_names: list[str],
        model_entries: list[tuple[str, str, int]] | None = None,
    ) -> None:
        self._worker_names = worker_names
        self._model_names = model_names
        self._model_entries = model_entries or []

        counts = self.query_one(StatusCounts)
        counts.running = running_count
        counts.registered = worker_count
        counts.memory_pct = mem_pct

        workers_t = self.query_one("#workers-table", DataTable)
        prev_w_cursor = workers_t.cursor_row
        workers_t.clear()
        for name, host, port, status, version, cpu, gpu, mem, models in worker_rows:
            workers_t.add_row(
                f"[bold]{name}[/]", host, port, status, version,
                cpu, gpu, mem, models,
            )
        if not worker_rows:
            workers_t.add_row(
                "[dim]—[/]", "—", "—", "—", "—", "—", "—", "—",
                "[dim]No workers registered[/]",
            )
        elif prev_w_cursor is not None:
            workers_t.move_cursor(row=min(prev_w_cursor, len(worker_rows) - 1))

        models_t = self.query_one("#cluster-models-table", DataTable)
        prev_m_cursor = models_t.cursor_row
        models_t.clear()
        for row in model_rows:
            models_t.add_row(*row)
        if not model_rows:
            models_t.add_row(
                "[dim]—[/]", "[dim]—[/]", "[dim]—[/]",
                "[dim]—[/]", "[dim]—[/]",
                "[dim]No models running[/]",
            )
        elif prev_m_cursor is not None:
            models_t.move_cursor(row=min(prev_m_cursor, len(model_rows) - 1))

    def _apply_no_head(self, message: str = "Start a node with --head") -> None:
        """Show empty state when no head node is reachable."""
        counts = self.query_one(StatusCounts)
        counts.running = 0
        counts.registered = 0
        counts.memory_pct = 0.0
        counts.memory_pressure = "unknown"

        workers_t = self.query_one("#workers-table", DataTable)
        workers_t.clear()
        workers_t.add_row(
            "[yellow]No head node[/]", "—", "—", "—", "—", "—", "—", "—",
            f"[dim]{message}[/]",
        )

        models_t = self.query_one("#cluster-models-table", DataTable)
        models_t.clear()

    # ── Register ──────────────────────────────────────

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

    # ── Spawn ─────────────────────────────────────────

    def action_spawn(self) -> None:
        """Show a modal to spawn a model on a worker."""
        if not self._worker_names:
            self.notify("No workers available", severity="warning")
            return

        from silo.tui.widgets.cluster_spawn_modal import ClusterSpawnModal

        def on_result(result: dict | None) -> None:
            if result is not None:
                self._do_spawn(result)

        head_url = self._get_head_url()
        self.app.push_screen(
            ClusterSpawnModal(self._worker_names, head_url=head_url),
            on_result,
        )

    @work(thread=True)
    def _do_spawn(self, data: dict) -> None:
        import json
        import urllib.request

        head_url = self._get_head_url()
        if head_url is None:
            return

        self.app.call_from_thread(
            self.notify,
            f"Spawning {data['name']} on {data['node']}...",
        )

        try:
            body = json.dumps(data).encode()
            req = urllib.request.Request(
                f"{head_url}/cluster/spawn",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            self.app.call_from_thread(
                self.notify,
                f"Spawned {result['name']} on {result['node']} (PID {result['pid']})",
            )
        except Exception as exc:
            self.app.call_from_thread(
                self.notify,
                f"Spawn failed: {exc}",
                severity="error",
            )

        self._load_data()

    # ── Stop ──────────────────────────────────────────

    def action_stop_model(self) -> None:
        """Stop the selected model from the cluster models table."""
        models_t = self.query_one("#cluster-models-table", DataTable)
        if models_t.cursor_row is None or models_t.row_count == 0:
            return
        if not self._model_names:
            return

        row_idx = models_t.cursor_row
        if row_idx >= len(self._model_names):
            return
        model_name = self._model_names[row_idx]

        from silo.tui.widgets.confirm_modal import ConfirmModal

        def on_confirm(confirmed: bool) -> None:
            if confirmed:
                self._do_stop(model_name)

        self.app.push_screen(
            ConfirmModal(f"Stop model '{model_name}'?"), on_confirm
        )

    @work(thread=True)
    def _do_stop(self, model_name: str) -> None:
        import json
        import urllib.request

        head_url = self._get_head_url()
        if head_url is None:
            return

        self.app.call_from_thread(
            self.notify, f"Stopping {model_name}..."
        )

        try:
            body = json.dumps({"name": model_name}).encode()
            req = urllib.request.Request(
                f"{head_url}/cluster/stop",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            self.app.call_from_thread(
                self.notify,
                f"Stopped {result['name']} on {result['node']}",
            )
        except Exception as exc:
            self.app.call_from_thread(
                self.notify,
                f"Stop failed: {exc}",
                severity="error",
            )

        self._load_data()

    # ── Download ──────────────────────────────────────

    def action_download(self) -> None:
        """Show a modal to download a model to a worker."""
        if not self._worker_names:
            self.notify("No workers available", severity="warning")
            return

        from silo.tui.widgets.cluster_download_modal import ClusterDownloadModal

        def on_result(result: dict | None) -> None:
            if result is not None:
                self._do_download(result)

        self.app.push_screen(
            ClusterDownloadModal(self._worker_names), on_result
        )

    @work(thread=True)
    def _do_download(self, data: dict) -> None:
        import json
        import urllib.request

        head_url = self._get_head_url()
        if head_url is None:
            return

        self.app.call_from_thread(
            self.notify,
            f"Downloading {data['repo_id']} to {data['node']}...",
        )

        try:
            body = json.dumps(data).encode()
            req = urllib.request.Request(
                f"{head_url}/cluster/download",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read())
            self.app.call_from_thread(
                self.notify,
                f"Downloaded {result['repo_id']} on {result['node']}",
            )
        except Exception as exc:
            self.app.call_from_thread(
                self.notify,
                f"Download failed: {exc}",
                severity="error",
            )

        self._load_data()

    # ── Remove ────────────────────────────────────────

    def action_remove_worker(self) -> None:
        """Remove the selected worker from the cluster."""
        workers_t = self.query_one("#workers-table", DataTable)
        if workers_t.cursor_row is None or workers_t.row_count == 0:
            return
        if not self._worker_names:
            return

        row_idx = workers_t.cursor_row
        if row_idx >= len(self._worker_names):
            return
        worker_name = self._worker_names[row_idx]

        from silo.tui.widgets.confirm_modal import ConfirmModal

        def on_confirm(confirmed: bool) -> None:
            if confirmed:
                self._do_remove(worker_name)

        self.app.push_screen(
            ConfirmModal(f"Remove worker '{worker_name}'?"), on_confirm
        )

    @work(thread=True)
    def _do_remove(self, worker_name: str) -> None:
        import urllib.request

        head_url = self._get_head_url()
        if head_url is None:
            return

        try:
            req = urllib.request.Request(
                f"{head_url}/cluster/workers/{worker_name}",
                method="DELETE",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
            self.app.call_from_thread(
                self.notify, f"Removed worker '{worker_name}'"
            )
        except Exception as exc:
            self.app.call_from_thread(
                self.notify,
                f"Remove failed: {exc}",
                severity="error",
            )

        self._load_data()

    # ── Edit ───────────────────────────────────────────

    def action_edit_model(self) -> None:
        """Edit the selected model from the cluster models table."""
        models_t = self.query_one("#cluster-models-table", DataTable)
        if models_t.cursor_row is None or models_t.row_count == 0:
            return
        if not self._model_entries:
            return

        row_idx = models_t.cursor_row
        if row_idx >= len(self._model_entries):
            return

        name, worker_host, model_port = self._model_entries[row_idx]
        if not worker_host or not model_port:
            self.notify("Cannot determine server address", severity="warning")
            return

        self._open_cluster_edit(name, worker_host, model_port)

    @work(thread=True)
    def _open_cluster_edit(self, name: str, worker_host: str, model_port: int) -> None:
        """Fetch admin info from the model server and open the edit modal."""
        import json
        import urllib.request

        server_url = f"http://{worker_host}:{model_port}"

        current_model_name = name
        litellm_registered = False
        litellm_url = ""
        litellm_api_key = ""

        try:
            req = urllib.request.Request(f"{server_url}/admin/info")
            with urllib.request.urlopen(req, timeout=3) as resp:
                info = json.loads(resp.read())
                current_model_name = info.get("model_name", name)
                ll = info.get("litellm", {})
                litellm_registered = ll.get("registered", False)
                litellm_url = ll.get("url", "")
        except Exception:
            pass

        if not litellm_api_key:
            try:
                from silo.config.loader import load_config

                config = load_config()
                litellm_api_key = config.litellm.api_key
                if not litellm_url:
                    litellm_url = config.litellm.url
            except Exception:
                pass

        from silo.tui.widgets.edit_server_modal import EditServerModal

        def on_edit(result) -> None:
            if result is not None:
                self._do_cluster_update(name, worker_host, model_port, result)

        self.app.call_from_thread(
            self.app.push_screen,
            EditServerModal(
                name=name,
                current_model_name=current_model_name,
                current_port=model_port,
                litellm_registered=litellm_registered,
                litellm_url=litellm_url,
                litellm_api_key=litellm_api_key,
            ),
            on_edit,
        )

    @work(thread=True)
    def _do_cluster_update(
        self, name: str, worker_host: str, model_port: int, update,
    ) -> None:
        """Apply edit modal changes to a model server in the cluster."""
        import json
        import urllib.request

        server_url = f"http://{worker_host}:{model_port}"
        changes: list[str] = []

        try:
            if update.litellm_enabled is True:
                from silo.litellm.registry import normalize_litellm_url

                data: dict[str, str] = {"url": normalize_litellm_url(update.litellm_url)}
                if update.litellm_api_key:
                    data["api_key"] = update.litellm_api_key
                body = json.dumps(data).encode()
                req = urllib.request.Request(
                    f"{server_url}/admin/litellm/register",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    json.loads(resp.read())
                changes.append("LiteLLM registered")

            elif update.litellm_enabled is False:
                body = json.dumps({}).encode()
                req = urllib.request.Request(
                    f"{server_url}/admin/litellm/deregister",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    json.loads(resp.read())
                changes.append("LiteLLM deregistered")

            if update.model_name:
                body = json.dumps({"model_name": update.model_name}).encode()
                req = urllib.request.Request(
                    f"{server_url}/admin/model-name",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="PUT",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    json.loads(resp.read())
                changes.append(f"renamed to '{update.model_name}'")

            if update.port:
                changes.append(f"port change to {update.port} requires local restart")

            if changes:
                self.app.call_from_thread(
                    self.notify, f"Updated {name}: {', '.join(changes)}"
                )
            else:
                self.app.call_from_thread(
                    self.notify, "No changes applied", severity="warning"
                )
        except Exception as exc:
            self.app.call_from_thread(
                self.notify, f"Update failed: {exc}", severity="error"
            )

        self._load_data()

    # ── Helpers ───────────────────────────────────────

    def _get_head_url(self) -> str | None:
        """Find the head node URL from config, app state, or mDNS discovery."""
        from silo.agent.client import resolve_head_url

        return resolve_head_url(self.app)
