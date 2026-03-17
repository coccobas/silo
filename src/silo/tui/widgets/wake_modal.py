"""Wake word configuration modal — pick flow, wake word, and threshold."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static


_BUILTIN_WAKE_WORDS: list[tuple[str, str]] = [
    ("Hey Jarvis", "hey_jarvis"),
    ("Alexa", "alexa"),
    ("Hey Mycroft", "hey_mycroft"),
    ("Ok Google", "ok_google"),
]


@dataclass(frozen=True)
class WakeSettings:
    """Collected wake word settings from the modal."""

    wake_word: str
    flow_name: str
    threshold: float
    continuous: bool
    device: int | None


class WakeModal(ModalScreen[WakeSettings | None]):
    """Modal dialog to configure and start wake word listening."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("tab", "focus_next", "Next Field"),
        ("shift+tab", "focus_previous", "Prev Field"),
    ]

    def __init__(self, flow_names: list[str]) -> None:
        super().__init__()
        self._flow_names = flow_names

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        flow_options: list[tuple[str, str]] = [
            (name, name) for name in self._flow_names
        ]

        with Vertical(id="wake-dialog"):
            yield Static(
                "[b]Wake Word Settings[/b]",
                id="wake-title",
            )

            # ── Wake word ──
            yield Static("[b dim]Detection[/]", classes="serve-section")
            with Horizontal(classes="form-row"):
                yield Label("Wake word:")
                yield Select(
                    _BUILTIN_WAKE_WORDS,
                    value="hey_jarvis",
                    id="wake-word",
                )
            with Horizontal(classes="form-row"):
                yield Label("Custom word:")
                yield Input(
                    value="",
                    placeholder="model name or .onnx path (overrides above)",
                    id="wake-custom-word",
                )
            with Horizontal(classes="form-row"):
                yield Label("Threshold:")
                yield Input(value="0.5", id="wake-threshold")

            # ── Flow ──
            yield Static("[b dim]Flow[/]", classes="serve-section")
            with Horizontal(classes="form-row"):
                yield Label("Flow:")
                if flow_options:
                    yield Select(
                        flow_options,
                        value=flow_options[0][1],
                        id="wake-flow",
                    )
                else:
                    yield Select(
                        [("(no flows found)", "")],
                        value="",
                        id="wake-flow",
                    )

            # ── Options ──
            yield Static("[b dim]Options[/]", classes="serve-section")
            with Horizontal(classes="form-row"):
                yield Label("")
                yield Checkbox(
                    "Continuous (keep listening after flow)",
                    id="wake-continuous",
                    value=True,
                )

            yield Static(
                "[dim]Tab: next field  Enter: open select / submit  Esc: cancel[/]",
                classes="serve-hint",
            )

            # ── Buttons ──
            with Horizontal(id="buttons"):
                yield Button("Start", variant="success", id="start")
                yield Button("Cancel", variant="primary", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#start", Button).focus()

    def _collect(self) -> WakeSettings | None:
        """Collect and validate all fields."""
        # Custom word overrides select
        custom = self.query_one("#wake-custom-word", Input).value.strip()
        if custom:
            wake_word = custom
        else:
            wake_word = str(self.query_one("#wake-word", Select).value)

        flow_name = str(self.query_one("#wake-flow", Select).value)
        if not flow_name:
            return None

        raw_threshold = self.query_one("#wake-threshold", Input).value.strip()
        try:
            threshold = float(raw_threshold) if raw_threshold else 0.5
        except ValueError:
            threshold = 0.5
        threshold = max(0.0, min(1.0, threshold))

        continuous = self.query_one("#wake-continuous", Checkbox).value

        return WakeSettings(
            wake_word=wake_word,
            flow_name=flow_name,
            threshold=threshold,
            continuous=continuous,
            device=None,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.dismiss(self._collect())
        else:
            self.dismiss(None)

    def on_key(self, event) -> None:
        focused = self.focused

        if event.key in ("left", "right", "up", "down"):
            if isinstance(focused, Button):
                start_btn = self.query_one("#start", Button)
                cancel_btn = self.query_one("#cancel", Button)
                if event.key in ("left", "up"):
                    (cancel_btn if focused is start_btn else start_btn).focus()
                else:
                    (start_btn if focused is cancel_btn else cancel_btn).focus()
                event.prevent_default()
                event.stop()
            return

        if event.key != "enter":
            return
        if focused is None:
            return
        if isinstance(focused, Select):
            focused.action_show_overlay()
            event.prevent_default()
            event.stop()
        elif isinstance(focused, Button) and focused.id == "start":
            self.dismiss(self._collect())
            event.prevent_default()
            event.stop()
        elif isinstance(focused, Button) and focused.id == "cancel":
            self.dismiss(None)
            event.prevent_default()
            event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_focus_next()
