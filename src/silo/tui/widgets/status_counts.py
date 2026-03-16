"""Status counts bar widget — shows running/registered/memory at a glance."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class StatusCounts(Static):
    """Compact bar showing running servers, registered models, and memory."""

    running: reactive[int] = reactive(0)
    registered: reactive[int] = reactive(0)
    memory_pct: reactive[float] = reactive(0.0)
    memory_pressure: reactive[str] = reactive("unknown")

    def render(self) -> str:
        pressure_indicator = {
            "normal": "[green]●[/]",
            "warn": "[yellow]●[/]",
            "critical": "[red]●[/]",
        }.get(self.memory_pressure, "[dim]●[/]")

        return (
            f"  Running: [bold]{self.running}[/]"
            f"  │  Registered: [bold]{self.registered}[/]"
            f"  │  Memory: {self.memory_pct:.0f}% {pressure_indicator}"
        )
