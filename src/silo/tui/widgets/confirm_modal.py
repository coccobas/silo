"""Reusable yes/no confirmation modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ConfirmModal(ModalScreen[bool]):
    """Modal dialog that returns True (confirm) or False (cancel)."""

    BINDINGS = [
        ("escape", "dismiss_no", "Cancel"),
    ]

    def __init__(self, question: str) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._question, id="question")
            with Horizontal(id="buttons"):
                yield Button("Yes", variant="error", id="yes")
                yield Button("No", variant="primary", id="no")

    def on_mount(self) -> None:
        self.query_one("#no", Button).focus()

    def on_key(self, event: Key) -> None:
        if event.key in ("left", "right"):
            yes_btn = self.query_one("#yes", Button)
            no_btn = self.query_one("#no", Button)
            if yes_btn.has_focus:
                no_btn.focus()
            else:
                yes_btn.focus()
            event.prevent_default()
            event.stop()
        elif event.key == "y":
            self.dismiss(True)
            event.prevent_default()
            event.stop()
        elif event.key == "n":
            self.dismiss(False)
            event.prevent_default()
            event.stop()

    def action_dismiss_no(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")
