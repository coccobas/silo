"""Flow creation modal — build a flow definition step-by-step."""

from __future__ import annotations

from dataclasses import dataclass, field

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static


_STEP_TYPES: list[tuple[str, str]] = [
    ("Text Generate (LLM)", "text.generate"),
    ("Audio Transcribe (STT)", "audio.transcribe"),
    ("File Glob", "fs.glob"),
    ("File Write", "fs.write"),
]


@dataclass(frozen=True)
class FlowStepDraft:
    """A step being built in the modal."""

    id: str
    type: str
    model: str
    input: str


@dataclass(frozen=True)
class FlowDraft:
    """Complete flow definition from the modal."""

    name: str
    description: str
    steps: list[FlowStepDraft] = field(default_factory=list)


def _needs_model(step_type: str) -> bool:
    return step_type in ("text.generate", "audio.transcribe")


class FlowCreateModal(ModalScreen[FlowDraft | None]):
    """Modal to create a new flow definition with multiple steps."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("tab", "focus_next", "Next Field"),
        ("shift+tab", "focus_previous", "Prev Field"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._steps: list[FlowStepDraft] = []

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        with Vertical(id="flow-create-dialog"):
            yield Static("[b]Create Flow[/b]", id="flow-create-title")

            with VerticalScroll(id="flow-create-scroll"):
                # ── Flow metadata ──
                yield Static("[b dim]Flow[/]", classes="serve-section")
                with Horizontal(classes="form-row"):
                    yield Label("Name:")
                    yield Input(
                        placeholder="my-flow (lowercase, hyphens)",
                        id="flow-name",
                    )
                with Horizontal(classes="form-row"):
                    yield Label("Description:")
                    yield Input(
                        placeholder="What does this flow do?",
                        id="flow-desc",
                    )

                # ── Steps list ──
                yield Static("[b dim]Steps[/]", classes="serve-section")
                yield Static(
                    "[dim]No steps added yet[/]",
                    id="steps-list",
                )

                # ── Add step form ──
                yield Static(
                    "[b dim]Add Step[/]",
                    classes="serve-section",
                    id="add-step-section",
                )
                with Horizontal(classes="form-row"):
                    yield Label("Step ID:")
                    yield Input(
                        placeholder="e.g. transcribe, summarize",
                        id="step-id",
                    )
                with Horizontal(classes="form-row"):
                    yield Label("Type:")
                    yield Select(
                        _STEP_TYPES,
                        value="text.generate",
                        id="step-type",
                    )
                with Horizontal(classes="form-row", id="step-model-row"):
                    yield Label("Model:")
                    yield Input(
                        placeholder="e.g. mlx-community/Llama-3.2-1B-4bit",
                        id="step-model",
                    )
                with Horizontal(classes="form-row"):
                    yield Label("Input:")
                    yield Input(
                        placeholder="$input, $steps.prev.output, or text",
                        id="step-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("")
                    yield Button(
                        "+ Add Step",
                        variant="success",
                        id="add-step",
                    )

            yield Static(
                "[dim]Tab: next field  Esc: cancel[/]",
                classes="serve-hint",
            )

            # ── Buttons ──
            with Horizontal(id="buttons"):
                yield Button("Save Flow", variant="success", id="save")
                yield Button("Cancel", variant="primary", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#flow-name", Input).focus()
        self._update_model_visibility()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "step-type":
            self._update_model_visibility()

    def _update_model_visibility(self) -> None:
        step_type = str(self.query_one("#step-type", Select).value)
        self.query_one("#step-model-row").display = _needs_model(step_type)

    def _render_steps_list(self) -> None:
        """Update the steps list display."""
        panel = self.query_one("#steps-list", Static)
        if not self._steps:
            panel.update("[dim]No steps added yet[/]")
            return

        lines: list[str] = []
        for i, step in enumerate(self._steps, 1):
            model_part = f"  model: {step.model}" if step.model else ""
            input_part = f"  input: {step.input}" if step.input else ""
            lines.append(
                f"  [bold]{i}.[/] [cyan]{step.id}[/] "
                f"[dim]({step.type})[/]"
                f"{model_part}{input_part}"
            )
        panel.update("\n".join(lines))

    def _add_step(self) -> None:
        """Validate and add the current step to the list."""
        step_id = self.query_one("#step-id", Input).value.strip()
        if not step_id:
            self.notify("Step ID is required", severity="error")
            return

        # Check for duplicate IDs
        if any(s.id == step_id for s in self._steps):
            self.notify(f"Step '{step_id}' already exists", severity="error")
            return

        step_type = str(self.query_one("#step-type", Select).value)
        model = self.query_one("#step-model", Input).value.strip()
        step_input = self.query_one("#step-input", Input).value.strip()

        if _needs_model(step_type) and not model:
            self.notify("This step type requires a model", severity="error")
            return

        step = FlowStepDraft(
            id=step_id,
            type=step_type,
            model=model,
            input=step_input,
        )
        self._steps = [*self._steps, step]
        self._render_steps_list()

        # Clear step form for next entry
        self.query_one("#step-id", Input).value = ""
        self.query_one("#step-model", Input).value = ""
        self.query_one("#step-input", Input).value = ""
        self.query_one("#step-id", Input).focus()

        self.notify(f"Step '{step_id}' added")

    def _collect(self) -> FlowDraft | None:
        """Collect and validate the flow definition."""
        name = self.query_one("#flow-name", Input).value.strip()
        if not name:
            self.notify("Flow name is required", severity="error")
            return None

        if not self._steps:
            self.notify("Add at least one step", severity="error")
            return None

        description = self.query_one("#flow-desc", Input).value.strip()

        return FlowDraft(
            name=name,
            description=description,
            steps=list(self._steps),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add-step":
            self._add_step()
        elif event.button.id == "save":
            result = self._collect()
            if result is not None:
                self.dismiss(result)
        elif event.button.id == "cancel":
            self.dismiss(None)

    def on_key(self, event) -> None:
        focused = self.focused

        if event.key in ("left", "right", "up", "down"):
            if isinstance(focused, Button):
                buttons = list(self.query(Button))
                try:
                    idx = buttons.index(focused)
                except ValueError:
                    return
                if event.key in ("left", "up"):
                    new_idx = (idx - 1) % len(buttons)
                else:
                    new_idx = (idx + 1) % len(buttons)
                buttons[new_idx].focus()
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
        elif isinstance(focused, Button) and focused.id == "add-step":
            self._add_step()
            event.prevent_default()
            event.stop()
        elif isinstance(focused, Button) and focused.id == "save":
            result = self._collect()
            if result is not None:
                self.dismiss(result)
            event.prevent_default()
            event.stop()
        elif isinstance(focused, Button) and focused.id == "cancel":
            self.dismiss(None)
            event.prevent_default()
            event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_focus_next()
