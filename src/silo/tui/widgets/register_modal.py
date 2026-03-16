"""Modal dialog for registering a new worker node with the cluster."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


class RegisterModal(ModalScreen[dict | None]):
    """Prompt for worker node details: name, host, port."""

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Register Worker Node", id="question")
            with Horizontal(classes="form-row"):
                yield Label("Name:")
                yield Input(placeholder="mini-1", id="reg-name")
            with Horizontal(classes="form-row"):
                yield Label("Host:")
                yield Input(placeholder="10.0.0.5", id="reg-host")
            with Horizontal(classes="form-row"):
                yield Label("Port:")
                yield Input(placeholder="9900", id="reg-port", value="9900")
            with Horizontal(id="buttons"):
                yield Button("Register", variant="primary", id="btn-register")
                yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-register":
            name = self.query_one("#reg-name", Input).value.strip()
            host = self.query_one("#reg-host", Input).value.strip()
            port_str = self.query_one("#reg-port", Input).value.strip()
            if name and host and port_str:
                try:
                    port = int(port_str)
                except ValueError:
                    return
                self.dismiss({"name": name, "host": host, "port": port})
            return
        self.dismiss(None)
