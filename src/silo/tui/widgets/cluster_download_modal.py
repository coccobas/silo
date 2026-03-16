"""Modal for downloading a model to a cluster worker node."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static


class ClusterDownloadModal(ModalScreen[dict | None]):
    """Collect download parameters: node and repo_id."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, worker_names: list[str]) -> None:
        super().__init__()
        self._worker_names = worker_names

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        options = [(name, name) for name in self._worker_names]
        with Vertical(id="dialog"):
            yield Static("Download Model to Worker", id="question")
            with Horizontal(classes="form-row"):
                yield Label("Node:")
                yield Select(
                    options,
                    value=options[0][1] if options else Select.BLANK,
                    id="dl-node",
                )
            with Horizontal(classes="form-row"):
                yield Label("Repo ID:")
                yield Input(
                    placeholder="mlx-community/Llama-3.2-1B-4bit",
                    id="dl-repo",
                )
            with Horizontal(id="buttons"):
                yield Button("Download", variant="success", id="btn-download")
                yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-download":
            node = self.query_one("#dl-node", Select).value
            repo_id = self.query_one("#dl-repo", Input).value.strip()
            if not repo_id or node is Select.BLANK:
                return
            self.dismiss({"node": str(node), "repo_id": repo_id})
            return
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_focus_next()
