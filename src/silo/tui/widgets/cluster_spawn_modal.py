"""Modal for spawning a model on a cluster worker node."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static


class ClusterSpawnModal(ModalScreen[dict | None]):
    """Collect spawn parameters: node, repo_id, name, host, port."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, worker_names: list[str]) -> None:
        super().__init__()
        self._worker_names = worker_names

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        options = [(name, name) for name in self._worker_names]
        with Vertical(id="dialog"):
            yield Static("Spawn Model on Worker", id="question")
            with Horizontal(classes="form-row"):
                yield Label("Node:")
                yield Select(
                    options,
                    value=options[0][1] if options else Select.BLANK,
                    id="spawn-node",
                )
            with Horizontal(classes="form-row"):
                yield Label("Repo ID:")
                yield Input(
                    placeholder="mlx-community/Llama-3.2-1B-4bit",
                    id="spawn-repo",
                )
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-spawn":
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
            return
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_focus_next()
