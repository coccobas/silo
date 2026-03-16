"""Download path selection modal."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


class DownloadModal(ModalScreen[str | None]):
    """Modal that asks for a download path. Returns the path or None on cancel."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def action_cancel(self) -> None:
        self.dismiss(None)

    DEFAULT_CSS = """
    DownloadModal {
        align: center middle;
    }
    DownloadModal > #dialog {
        width: 70;
        height: auto;
        max-height: 14;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    DownloadModal > #dialog > #title {
        width: 100%;
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    DownloadModal > #dialog > #hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    DownloadModal > #dialog > #buttons {
        width: 100%;
        height: auto;
        align-horizontal: center;
        margin-top: 1;
    }
    DownloadModal > #dialog > #buttons > Button {
        margin: 0 1;
    }
    """

    def __init__(self, repo_id: str) -> None:
        super().__init__()
        self._repo_id = repo_id
        self._default_path = str(
            Path.home() / "models" / repo_id.replace("/", "--")
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"Download {self._repo_id}", id="title")
            yield Static(
                "Leave blank for default HF cache (~/.cache/huggingface/hub)",
                id="hint",
            )
            yield Input(
                placeholder=self._default_path,
                id="download-path",
            )
            from textual.containers import Horizontal

            with Horizontal(id="buttons"):
                yield Button("Download", variant="success", id="download")
                yield Button("Cancel", variant="default", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "download":
            path = self.query_one("#download-path", Input).value.strip()
            # "" means default HF cache, non-empty means custom path
            self.dismiss(path)
        else:
            self.dismiss(None)  # cancelled

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())
