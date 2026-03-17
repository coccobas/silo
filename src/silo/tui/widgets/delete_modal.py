"""Delete confirmation modal with option to remove files."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class DeleteModal(ModalScreen[str | None]):
    """Modal that returns 'registry', 'files', or None (cancel)."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, repo_id: str, local_path: str | None = None) -> None:
        super().__init__()
        self._repo_id = repo_id
        self._local_path = local_path

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(f"Delete '{self._repo_id}'?", id="question")
            if self._local_path:
                yield Static(
                    f"[dim]{self._local_path}[/]", classes="serve-hint"
                )
            with Horizontal(id="buttons"):
                yield Button(
                    "Registry only", variant="primary", id="btn-registry"
                )
                if self._local_path:
                    yield Button(
                        "Registry + files",
                        variant="error",
                        id="btn-files",
                    )
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        self.query_one("#btn-cancel", Button).focus()

    def on_key(self, event: Key) -> None:
        if event.key in ("left", "right"):
            buttons = list(self.query("Button"))
            focused = [b for b in buttons if b.has_focus]
            if focused:
                idx = buttons.index(focused[0])
                next_idx = (idx + (1 if event.key == "right" else -1)) % len(
                    buttons
                )
                buttons[next_idx].focus()
            event.prevent_default()
            event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-registry":
            self.dismiss("registry")
        elif event.button.id == "btn-files":
            self.dismiss("files")
        else:
            self.dismiss(None)
