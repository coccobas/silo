"""Models screen — local registry, HF search, conversion."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from silo.tui.widgets.confirm_modal import ConfirmModal
from silo.tui.widgets.download_modal import DownloadModal
from silo.tui.widgets.nav_bar import NavBar
from silo.tui.widgets.serve_modal import ServeModal, ServeSettings


def _fmt_size(size_bytes: int | None) -> str:
    """Format byte count as human-readable size."""
    if size_bytes is None:
        return "—"
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.0f} MB"
    return f"{size_bytes / 1024:.0f} KB"


class ModelsScreen(Screen):
    """Model management: local registry, search, convert."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("slash", "focus_search", "Search"),
        ("enter", "serve_selected", "Serve"),
        ("delete", "delete_selected", "Delete"),
        ("ctrl+d", "download_selected", "Download"),
        ("n", "next_page", "Next Page"),
        ("p", "prev_page", "Prev Page"),
    ]

    _search_query: str = ""
    _search_mlx_only: bool = True
    _search_page: int = 0
    _search_page_size: int = 20
    _search_has_more: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(id="models-tabs"):
            with TabPane("Local", id="tab-local"):
                yield DataTable(id="local-table")
                with Horizontal(classes="form-row"):
                    yield Button("Serve", id="serve-btn", variant="success")
                    yield Button("Delete", id="delete-btn", variant="error")
                yield Static(id="local-detail", classes="detail-panel")
            with TabPane("Search", id="tab-search"):
                with Horizontal(classes="form-row"):
                    yield Input(
                        placeholder="Search HuggingFace models...",
                        id="search-input",
                    )
                    yield Checkbox("MLX only", id="mlx-checkbox", value=True)
                    yield Button("Search", id="search-btn", variant="primary")
                    yield Button("Download", id="download-btn", variant="success")
                yield DataTable(id="search-table")
                with Horizontal(classes="page-bar"):
                    yield Button("< Prev", id="prev-page-btn", variant="default")
                    yield Static("", id="page-indicator")
                    yield Button("Next >", id="next-page-btn", variant="default")
                yield Static(id="search-detail", classes="detail-panel")
            with TabPane("Convert", id="tab-convert"):
                with Vertical():
                    with Horizontal(classes="form-row"):
                        yield Label("Repo ID:")
                        yield Input(
                            placeholder="e.g. mlx-community/Llama-3.2-1B",
                            id="convert-repo",
                        )
                    with Horizontal(classes="form-row"):
                        yield Label("Quantize:")
                        yield Select(
                            [
                                ("None", "none"),
                                ("q2 (2-bit)", "q2"),
                                ("q3 (3-bit)", "q3"),
                                ("q4 (4-bit)", "q4"),
                                ("q6 (6-bit)", "q6"),
                                ("q8 (8-bit)", "q8"),
                            ],
                            value="q4",
                            id="convert-quant",
                        )
                    with Horizontal(classes="form-row"):
                        yield Label("Output:")
                        yield Input(
                            placeholder="Leave blank for default ~/models/...",
                            id="convert-output",
                        )
                    yield Button(
                        "Convert", id="convert-btn", variant="success"
                    )
                    yield Static(id="convert-status")
        yield NavBar(active_screen="models")
        yield Footer()

    def on_key(self, event) -> None:
        """Arrow down from tabs focuses into the active tab content."""
        if event.key != "down":
            return
        # Only intercept if nothing inside a tab is focused yet
        focused = self.focused
        if focused is not None and not hasattr(focused, "active"):
            # Already focused on a widget inside a tab — let default handle it
            return
        tabs = self.query_one("#models-tabs", TabbedContent)
        active = tabs.active
        if active == "tab-local":
            self.query_one("#local-table", DataTable).focus()
        elif active == "tab-search":
            self.query_one("#search-input", Input).focus()
        elif active == "tab-convert":
            self.query_one("#convert-repo", Input).focus()
        event.prevent_default()
        event.stop()

    def on_mount(self) -> None:
        local = self.query_one("#local-table", DataTable)
        local.add_columns("REPO ID", "FORMAT", "SIZE", "DOWNLOADED")
        local.cursor_type = "row"

        search = self.query_one("#search-table", DataTable)
        search.add_columns("REPO ID", "SIZE", "DOWNLOADS", "LIKES", "PIPELINE")
        search.cursor_type = "row"

        self._load_local()

    def action_refresh(self) -> None:
        self._load_local()

    def action_focus_search(self) -> None:
        """Switch to Search tab and focus the search input."""
        tabs = self.query_one("#models-tabs", TabbedContent)
        tabs.active = "tab-search"
        self.query_one("#search-input", Input).focus()

    # ── Local tab ────────────────────────────────────────

    @work(thread=True)
    def _load_local(self) -> None:
        from silo.registry.store import Registry

        registry = Registry.load()
        entries = registry.list()
        rows = []
        for entry in entries:
            size = (
                f"{entry.size_bytes / 1_073_741_824:.1f} GB"
                if entry.size_bytes
                else "—"
            )
            rows.append((
                entry.repo_id,
                entry.format,
                size,
                entry.downloaded_at[:10] if entry.downloaded_at else "—",
            ))
        self.app.call_from_thread(self._apply_local, rows)

    def _apply_local(self, rows) -> None:
        table = self.query_one("#local-table", DataTable)
        table.clear()
        if rows:
            for repo_id, fmt, size, date in rows:
                table.add_row(repo_id, str(fmt), size, date)
        else:
            table.add_row("[dim]No models registered[/]", "—", "—", "—")

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        table_id = event.data_table.id
        if table_id == "local-table":
            self._show_local_detail(event)
        elif table_id == "search-table":
            self._show_search_detail(event)

    def _show_local_detail(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None:
            return
        table = self.query_one("#local-table", DataTable)
        row_idx = list(table.rows.keys()).index(event.row_key)
        repo_id = str(table.get_cell_at((row_idx, 0)))

        from silo.registry.store import Registry

        entry = Registry.load().get(repo_id)
        if entry:
            detail = (
                f"[b]{entry.repo_id}[/b]\n"
                f"Format: {entry.format}  │  "
                f"Path: {entry.local_path or '—'}  │  "
                f"Tags: {', '.join(entry.tags) or '—'}"
            )
        else:
            detail = ""
        self.query_one("#local-detail", Static).update(detail)

    def action_serve_selected(self) -> None:
        """Serve the selected local model."""
        tabs = self.query_one("#models-tabs", TabbedContent)
        if tabs.active != "tab-local":
            return
        table = self.query_one("#local-table", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            return
        repo_id = str(table.get_cell_at((table.cursor_row, 0)))
        if repo_id.startswith("["):
            return  # placeholder row

        from silo.registry.store import Registry

        entry = Registry.load().get(repo_id)
        model_format = str(entry.format) if entry else "unknown"

        def on_serve(settings: ServeSettings | None) -> None:
            if settings is None:
                return
            self._do_serve(repo_id, settings)

        self.app.push_screen(
            ServeModal(repo_id, model_format=model_format), on_serve
        )

    @work(thread=True)
    def _do_serve(self, repo_id: str, settings: ServeSettings) -> None:
        import time

        from silo.config.paths import LOGS_DIR
        from silo.process.manager import spawn_model
        from silo.process.pid import is_running

        self.app.call_from_thread(
            self.notify,
            f"Starting {settings.name} ({settings.runtime}) "
            f"on {settings.host}:{settings.port}...",
        )
        try:
            pid = spawn_model(
                name=settings.name,
                repo_id=repo_id,
                host=settings.host,
                port=settings.port,
            )

            # Wait briefly and check if process is still alive
            time.sleep(2)
            if not is_running(pid):
                # Process died — read log for error details
                log_file = LOGS_DIR / f"{settings.name}.log"
                error_msg = "Process exited immediately"
                try:
                    lines = log_file.read_text().strip().splitlines()
                    # Show last few meaningful lines
                    tail = [line for line in lines[-5:] if line.strip()]
                    if tail:
                        error_msg = tail[-1]
                except Exception:
                    pass
                self.app.call_from_thread(
                    self.notify,
                    f"Failed to start {settings.name}: {error_msg}",
                    severity="error",
                )
                return

            self.app.call_from_thread(
                self.notify,
                f"Started {settings.name} (PID {pid}) on port {settings.port}",
            )
            if settings.warmup:
                self._warmup(settings)
        except Exception as exc:
            self.app.call_from_thread(
                self.notify, f"Failed to start: {exc}", severity="error"
            )

    def _warmup(self, settings: ServeSettings) -> None:
        """Send a warmup request to pre-load the model into memory."""
        import json
        import time
        import urllib.request

        url = f"http://{settings.host}:{settings.port}/v1/chat/completions"
        body = json.dumps({
            "model": settings.name,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
        }).encode()

        # Wait for server to be ready
        for _ in range(30):
            try:
                health_url = f"http://{settings.host}:{settings.port}/health"
                urllib.request.urlopen(health_url, timeout=2)
                break
            except Exception:
                time.sleep(1)

        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=120)
            self.app.call_from_thread(
                self.notify, f"Warmup complete for {settings.name}"
            )
        except Exception as exc:
            self.app.call_from_thread(
                self.notify, f"Warmup failed: {exc}", severity="warning"
            )

    def action_delete_selected(self) -> None:
        table = self.query_one("#local-table", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            return
        repo_id = str(table.get_cell_at((table.cursor_row, 0)))
        if repo_id.startswith("["):
            return  # placeholder row

        def on_confirm(confirmed: bool) -> None:
            if confirmed:
                self._delete_model(repo_id)

        self.app.push_screen(
            ConfirmModal(f"Delete '{repo_id}' from registry?"),
            on_confirm,
        )

    @work(thread=True)
    def _delete_model(self, repo_id: str) -> None:
        from silo.registry.store import Registry

        registry = Registry.load()
        updated = registry.remove(repo_id)
        updated.save()
        self.app.call_from_thread(
            self.notify, f"Removed {repo_id} from registry"
        )
        self.app.call_from_thread(self._load_local)

    # ── Search tab ───────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "serve-btn":
            self.action_serve_selected()
        elif event.button.id == "delete-btn":
            self.action_delete_selected()
        elif event.button.id == "search-btn":
            self._search_page = 0
            self._start_search()
        elif event.button.id == "download-btn":
            self.action_download_selected()
        elif event.button.id == "convert-btn":
            self._do_convert()
        elif event.button.id == "next-page-btn":
            self.action_next_page()
        elif event.button.id == "prev-page-btn":
            self.action_prev_page()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-input":
            self._search_page = 0
            self._start_search()

    def _start_search(self) -> None:
        """Capture query and kick off the search worker."""
        self._search_query = self.query_one("#search-input", Input).value.strip()
        self._search_mlx_only = self.query_one("#mlx-checkbox", Checkbox).value
        if self._search_query:
            self._do_search()

    def action_next_page(self) -> None:
        tabs = self.query_one("#models-tabs", TabbedContent)
        if tabs.active != "tab-search" or not self._search_query:
            return
        if self._search_has_more:
            self._search_page += 1
            self._do_search()

    def action_prev_page(self) -> None:
        tabs = self.query_one("#models-tabs", TabbedContent)
        if tabs.active != "tab-search" or not self._search_query:
            return
        if self._search_page > 0:
            self._search_page -= 1
            self._do_search()

    @work(thread=True)
    def _do_search(self) -> None:
        from silo.download.hf import search_models

        # Fetch one extra to detect if there's a next page
        fetch_limit = self._search_page_size + 1
        offset = self._search_page * self._search_page_size

        try:
            results = search_models(
                self._search_query,
                mlx_only=self._search_mlx_only,
                limit=fetch_limit,
                offset=offset,
            )
            has_more = len(results) > self._search_page_size
            page_results = results[: self._search_page_size]
            rows = [
                (
                    r["id"],
                    _fmt_size(r.get("size_bytes")),
                    str(r.get("downloads", 0)),
                    str(r.get("likes", 0)),
                    r.get("pipeline_tag", "—") or "—",
                )
                for r in page_results
            ]
            self.app.call_from_thread(self._apply_search, rows, has_more)
        except Exception as exc:
            self.app.call_from_thread(
                self.notify, f"Search failed: {exc}", severity="error"
            )

    def _apply_search(self, rows, has_more: bool = False) -> None:
        self._search_has_more = has_more

        table = self.query_one("#search-table", DataTable)
        table.clear()
        for repo_id, size, downloads, likes, pipeline in rows:
            table.add_row(repo_id, size, downloads, likes, pipeline)
        if not rows:
            table.add_row("[dim]No results[/]", "—", "—", "—", "—")

        # Update page indicator
        page_num = self._search_page + 1
        indicator = self.query_one("#page-indicator", Static)
        indicator.update(f" Page {page_num} ")

        # Enable/disable page buttons
        self.query_one("#prev-page-btn", Button).disabled = self._search_page == 0
        self.query_one("#next-page-btn", Button).disabled = not has_more

        # Focus the table so user can scroll with arrow keys
        table.focus()

    def _show_search_detail(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None:
            return
        table = self.query_one("#search-table", DataTable)
        row_idx = list(table.rows.keys()).index(event.row_key)
        repo_id = str(table.get_cell_at((row_idx, 0)))
        if repo_id.startswith("["):
            return
        self._fetch_model_info(repo_id)

    @work(thread=True)
    def _fetch_model_info(self, repo_id: str) -> None:
        from silo.download.hf import get_model_info

        try:
            info = get_model_info(repo_id)
            detail = (
                f"[b]{info['id']}[/b] by {info.get('author', '—')}\n"
                f"Downloads: {info.get('downloads', 0):,}  │  "
                f"Likes: {info.get('likes', 0):,}  │  "
                f"Pipeline: {info.get('pipeline_tag', '—')}  │  "
                f"Library: {info.get('library_name', '—')}\n"
                f"Tags: {', '.join(info.get('tags', [])[:8])}"
            )
            self.app.call_from_thread(
                self.query_one("#search-detail", Static).update, detail
            )
        except Exception:
            pass

    # ── Download from search ────────────────────────────

    def action_download_selected(self) -> None:
        """Download the selected model from the search results."""
        try:
            search_table = self.query_one("#search-table", DataTable)
        except Exception:
            return
        if search_table.cursor_row is None or search_table.row_count == 0:
            return
        # Only act when search tab is active
        tabs = self.query_one("#models-tabs", TabbedContent)
        if tabs.active != "tab-search":
            return
        repo_id = str(search_table.get_cell_at((search_table.cursor_row, 0)))
        if repo_id.startswith("["):
            return

        def on_path_selected(local_dir: str | None) -> None:
            if local_dir is None:
                return  # cancelled
            # "" means default HF cache, non-empty means custom path
            self._do_download(repo_id, local_dir or None)

        self.app.push_screen(
            DownloadModal(repo_id),
            on_path_selected,
        )

    @work(thread=True)
    def _do_download(self, repo_id: str, local_dir: str | None = None) -> None:
        from silo.download.hf import download_model, get_model_info
        from silo.registry.detector import detect_model_format
        from silo.registry.models import RegistryEntry
        from silo.registry.store import Registry

        tracker = self.app.downloads
        # Get model info first to know total size
        try:
            info = get_model_info(repo_id)
            total_bytes = sum(
                s.get("size", 0) for s in info.get("siblings", [])
            ) if info.get("siblings") else 0
        except Exception:
            info = None
            total_bytes = 0

        tracker.start(repo_id, node="local", total_bytes=total_bytes)
        self.app.call_from_thread(
            self.notify, f"Downloading {repo_id}..."
        )

        try:
            path = download_model(repo_id, local_dir=local_dir)
            if info is None:
                info = get_model_info(repo_id)
            fmt = detect_model_format(repo_id, info.get("siblings"))
            entry = RegistryEntry(
                repo_id=repo_id,
                format=fmt,
                local_path=str(path),
            )
            registry = Registry.load()
            updated = registry.add(entry)
            updated.save()

            tracker.complete(repo_id, str(path))
            self.app.call_from_thread(
                self.notify, f"Downloaded {repo_id} → {path}"
            )
            self.app.call_from_thread(self._load_local)
        except Exception as exc:
            tracker.fail(repo_id, str(exc))
            self.app.call_from_thread(
                self.notify, f"Download failed: {exc}", severity="error"
            )

    # ── Convert tab ──────────────────────────────────────

    @work(thread=True)
    def _do_convert(self) -> None:
        repo_id = self.query_one("#convert-repo", Input).value.strip()
        if not repo_id:
            self.app.call_from_thread(
                self.notify, "Repo ID is required", severity="warning"
            )
            return

        quant_val = self.query_one("#convert-quant", Select).value
        quantize = None if quant_val == "none" else str(quant_val)
        output = self.query_one("#convert-output", Input).value.strip() or None

        self.app.call_from_thread(
            self.query_one("#convert-status", Static).update,
            f"[yellow]Converting {repo_id}...[/]",
        )

        from silo.convert.mlx import convert_model

        try:
            path = convert_model(
                repo_id=repo_id, quantize=quantize, output=output
            )
            self.app.call_from_thread(
                self.query_one("#convert-status", Static).update,
                f"[green]Done → {path}[/]",
            )
            self.app.call_from_thread(
                self.notify, f"Converted {repo_id}"
            )
        except Exception as exc:
            self.app.call_from_thread(
                self.query_one("#convert-status", Static).update,
                f"[red]Error: {exc}[/]",
            )
            self.app.call_from_thread(
                self.notify, f"Conversion failed: {exc}", severity="error"
            )
