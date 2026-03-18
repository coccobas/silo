"""Lightweight edit modal for a running model server."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Static


@dataclass(frozen=True)
class ServerUpdate:
    """Collected changes from the edit modal."""

    litellm_enabled: bool | None
    litellm_url: str
    litellm_api_key: str
    model_name: str
    port: int | None


class EditServerModal(ModalScreen[ServerUpdate | None]):
    """Edit settings on an already-running model server.

    Pre-populated from the server's current state via /admin/info.
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        name: str,
        current_model_name: str,
        current_port: int,
        litellm_registered: bool = False,
        litellm_url: str = "",
        litellm_api_key: str = "",
    ) -> None:
        super().__init__()
        self._name = name
        self._current_model_name = current_model_name
        self._current_port = current_port
        self._litellm_registered = litellm_registered
        self._litellm_url = litellm_url
        self._litellm_api_key = litellm_api_key

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(
                f"[b]Edit Server[/b]  —  {self._name}",
                id="question",
            )

            # Model name
            yield Static("[b dim]Identity[/]", classes="serve-section")
            with Horizontal(classes="form-row"):
                yield Label("Name:")
                yield Input(
                    value=self._current_model_name,
                    id="edit-model-name",
                )

            # Port (cold change)
            with Horizontal(classes="form-row"):
                yield Label("Port:")
                yield Input(
                    value=str(self._current_port),
                    id="edit-port",
                )
            yield Static(
                "[dim]Changing port will restart the server[/]",
                classes="serve-hint",
            )

            # LiteLLM
            yield Static("[b dim]LiteLLM Proxy[/]", classes="serve-section")
            with Horizontal(classes="form-row"):
                yield Label("")
                yield Checkbox(
                    "Register with LiteLLM",
                    id="edit-litellm-register",
                    value=self._litellm_registered,
                )
            with Horizontal(classes="form-row"):
                yield Label("URL:")
                yield Input(
                    value=self._litellm_url,
                    placeholder="e.g. 100.112.188.75",
                    id="edit-litellm-url",
                )
            with Horizontal(classes="form-row"):
                yield Label("API Key:")
                yield Input(
                    value=self._litellm_api_key,
                    placeholder="sk-...",
                    id="edit-litellm-key",
                    password=True,
                )

            with Horizontal(id="buttons"):
                yield Button("Apply", variant="success", id="btn-apply")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        self.query_one("#btn-apply", Button).focus()

    def _collect(self) -> ServerUpdate | None:
        model_name = self.query_one("#edit-model-name", Input).value.strip()
        port_str = self.query_one("#edit-port", Input).value.strip()
        litellm_reg = self.query_one("#edit-litellm-register", Checkbox).value
        litellm_url = self.query_one("#edit-litellm-url", Input).value.strip()
        litellm_key = self.query_one("#edit-litellm-key", Input).value.strip()

        try:
            port_val = int(port_str) if port_str else self._current_port
        except ValueError:
            port_val = self._current_port

        # Determine what changed
        litellm_enabled: bool | None = None
        if litellm_reg != self._litellm_registered:
            litellm_enabled = litellm_reg
        elif litellm_reg and (litellm_url != self._litellm_url):
            # URL changed while still enabled — re-register
            litellm_enabled = True

        port_changed = port_val if port_val != self._current_port else None
        name_changed = model_name if model_name != self._current_model_name else ""

        if litellm_enabled is None and not port_changed and not name_changed:
            return None  # No changes

        return ServerUpdate(
            litellm_enabled=litellm_enabled,
            litellm_url=litellm_url,
            litellm_api_key=litellm_key,
            model_name=name_changed,
            port=port_changed,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-apply":
            self.dismiss(self._collect())
        else:
            self.dismiss(None)
