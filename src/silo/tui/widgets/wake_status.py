"""Wake word status widget — shows listener state on the dashboard."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class WakeStatusBar(Static):
    """Compact bar showing wake word listener state."""

    state: reactive[str] = reactive("off")
    wake_word: reactive[str] = reactive("")
    flow_name: reactive[str] = reactive("")
    detections: reactive[int] = reactive(0)
    error: reactive[str] = reactive("")

    def render(self) -> str:
        if self.state == "off":
            return "  Wake: [dim]OFF[/dim]"

        indicator = {
            "listening": "[green]●[/] Listening",
            "detected": "[bold yellow]●[/] Detected!",
            "running_flow": "[cyan]●[/] Running flow",
            "error": "[red]●[/] Error",
            "stopped": "[dim]●[/] Stopped",
        }.get(self.state, f"[dim]●[/] {self.state}")

        parts = [f"  Wake: {indicator}"]

        if self.wake_word:
            parts.append(f"[dim]word:[/] {self.wake_word}")

        if self.state == "running_flow" and self.flow_name:
            parts.append(f"[dim]flow:[/] {self.flow_name}")

        if self.detections > 0:
            parts.append(f"[dim]detections:[/] {self.detections}")

        if self.state == "error" and self.error:
            parts.append(f"[red]{self.error}[/]")

        return "  │  ".join(parts)
