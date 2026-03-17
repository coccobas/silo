"""Modal for spawning a model on a cluster worker node."""

from __future__ import annotations

import json
import urllib.request

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, Select, Static


class ClusterSpawnModal(ModalScreen[dict | None]):
    """Spawn a model on a worker — browse local models, search HF, or enter manually."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self, worker_names: list[str], head_url: str | None = None
    ) -> None:
        super().__init__()
        self._worker_names = worker_names
        self._head_url = head_url
        self._selected_repo: str | None = None

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        options = [(name, name) for name in self._worker_names]
        with Vertical(id="dialog"):
            yield Static("[b]Spawn Model on Worker[/b]", id="question")

            with Horizontal(classes="form-row"):
                yield Label("Node:")
                yield Select(
                    options,
                    value=options[0][1] if options else Select.BLANK,
                    id="spawn-node",
                )

            # ── Model source ──
            yield Static("[b dim]Select Model[/]", classes="serve-section")
            with Horizontal(classes="form-row"):
                yield Button(
                    "Local Models", variant="primary", id="btn-load-local"
                )
                yield Button("Search HF", variant="default", id="btn-search")
            with Horizontal(classes="form-row"):
                yield Input(
                    placeholder="Search HuggingFace...",
                    id="spawn-search-input",
                )
            yield DataTable(id="spawn-models-table")

            with Horizontal(classes="form-row"):
                yield Label("Repo ID:")
                yield Input(
                    placeholder="or enter manually",
                    id="spawn-repo",
                )

            # ── Server settings ──
            yield Static("[b dim]Server Settings[/]", classes="serve-section")
            with Horizontal(classes="form-row"):
                yield Label("Name:")
                yield Input(placeholder="llama-1b", id="spawn-name")
            with Horizontal(classes="form-row"):
                yield Label("Host:")
                yield Input(value="0.0.0.0", id="spawn-host")
            with Horizontal(classes="form-row"):
                yield Label("Port:")
                yield Input(value="8800", id="spawn-port")

            with Horizontal(id="buttons"):
                yield Button("Spawn", variant="success", id="btn-spawn")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        table = self.query_one("#spawn-models-table", DataTable)
        table.add_columns("REPO ID", "FORMAT", "SIZE")
        table.cursor_type = "row"
        # Hide search input initially
        self.query_one("#spawn-search-input").display = False
        # Load local models for the first node
        self._load_local_models()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "spawn-node":
            self._load_local_models()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-load-local":
            self.query_one("#spawn-search-input").display = False
            self._load_local_models()
        elif event.button.id == "btn-search":
            self.query_one("#spawn-search-input").display = True
            self.query_one("#spawn-search-input", Input).focus()
        elif event.button.id == "btn-spawn":
            self._submit()
        elif event.button.id == "btn-cancel":
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "spawn-search-input":
            query = event.value.strip()
            if query:
                self._search_hf(query)
        else:
            self.action_focus_next()

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        if event.data_table.id != "spawn-models-table":
            return
        if event.row_key is None:
            return
        table = self.query_one("#spawn-models-table", DataTable)
        keys = list(table.rows.keys())
        if event.row_key not in keys:
            return
        row_idx = keys.index(event.row_key)
        repo_id = str(table.get_cell_at((row_idx, 0)))
        if repo_id.startswith("["):
            return
        self._selected_repo = repo_id
        repo_input = self.query_one("#spawn-repo", Input)
        repo_input.value = repo_id
        # Auto-fill name from repo
        name_input = self.query_one("#spawn-name", Input)
        if not name_input.value.strip():
            name_input.value = repo_id.split("/")[-1].lower().replace(" ", "-")

    @work(thread=True)
    def _load_local_models(self) -> None:
        """Fetch the selected worker's local model registry."""
        node = self.query_one("#spawn-node", Select).value
        if node is Select.BLANK or self._head_url is None:
            return

        # Get worker host/port from cluster status
        try:
            url = f"{self._head_url}/cluster/status"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except Exception:
            self.app.call_from_thread(self._apply_models, [], "Cannot reach head")
            return

        worker_host = None
        worker_port = None
        for w in data.get("workers", []):
            if w["name"] == str(node):
                worker_host = w["host"]
                worker_port = w["port"]
                break

        if worker_host is None:
            self.app.call_from_thread(self._apply_models, [], "Worker not found")
            return

        # Fetch the worker's registry directly
        try:
            reg_url = f"http://{worker_host}:{worker_port}/registry"
            req = urllib.request.Request(reg_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                registry = json.loads(resp.read())
        except Exception:
            self.app.call_from_thread(
                self._apply_models, [], "Cannot reach worker"
            )
            return

        rows = []
        for entry in registry:
            size_bytes = entry.get("size_bytes")
            if size_bytes and size_bytes >= 1_073_741_824:
                size = f"{size_bytes / 1_073_741_824:.1f} GB"
            elif size_bytes and size_bytes >= 1_048_576:
                size = f"{size_bytes / 1_048_576:.0f} MB"
            elif size_bytes:
                size = f"{size_bytes / 1024:.0f} KB"
            else:
                size = "—"
            rows.append((entry["repo_id"], entry.get("format", "—"), size))

        self.app.call_from_thread(self._apply_models, rows, None)

    @work(thread=True)
    def _search_hf(self, query: str) -> None:
        """Search HuggingFace models."""
        try:
            from silo.download.hf import search_models

            results = search_models(query, mlx_only=True, limit=10)
            rows = []
            for r in results:
                size_bytes = r.get("size_bytes")
                if size_bytes and size_bytes >= 1_073_741_824:
                    size = f"{size_bytes / 1_073_741_824:.1f} GB"
                elif size_bytes and size_bytes >= 1_048_576:
                    size = f"{size_bytes / 1_048_576:.0f} MB"
                else:
                    size = "—"
                rows.append((r["id"], r.get("pipeline_tag", "—"), size))
            self.app.call_from_thread(self._apply_models, rows, None)
        except Exception as exc:
            self.app.call_from_thread(
                self._apply_models, [], f"Search failed: {exc}"
            )

    def _apply_models(
        self, rows: list[tuple[str, str, str]], error: str | None
    ) -> None:
        table = self.query_one("#spawn-models-table", DataTable)
        table.clear()
        if error:
            table.add_row(f"[red]{error}[/]", "—", "—")
        elif rows:
            for repo_id, fmt, size in rows:
                table.add_row(repo_id, fmt, size)
        else:
            table.add_row("[dim]No models found[/]", "—", "—")

    def _submit(self) -> None:
        node = self.query_one("#spawn-node", Select).value
        repo_id = self.query_one("#spawn-repo", Input).value.strip()
        name = self.query_one("#spawn-name", Input).value.strip()
        host = self.query_one("#spawn-host", Input).value.strip()
        port_str = self.query_one("#spawn-port", Input).value.strip()
        if not repo_id or not name or node is Select.BLANK:
            return
        try:
            port = int(port_str)
        except ValueError:
            return
        self.dismiss({
            "node": str(node),
            "repo_id": repo_id,
            "name": name,
            "host": host or "0.0.0.0",
            "port": port,
        })
