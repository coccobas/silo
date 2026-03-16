"""Persistent navigation bar showing global screen shortcuts."""

from __future__ import annotations

from rich.text import Text
from textual.events import Click
from textual.widget import Widget


_NAV_ITEMS = [
    ("1", "Dashboard", "dashboard"),
    ("2", "Servers", "servers"),
    ("3", "Models", "models"),
    ("4", "Flows", "flows"),
    ("5", "Cluster", "cluster"),
    ("6", "Doctor", "doctor"),
]


class NavBar(Widget):
    """Compact single-line navigation bar. Clickable regions for each item."""

    DEFAULT_CSS = """
    NavBar {
        height: 1;
        width: 100%;
        background: $surface;
        padding: 0 1;
    }
    """

    def __init__(self, active_screen: str = "dashboard", **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._active_screen = active_screen
        self._regions: list[tuple[int, int, str]] = []  # (start_col, end_col, mode)

    def render(self) -> Text:
        result = Text()
        col = 0
        self._regions = []
        for key, label, mode in _NAV_ITEMS:
            segment = f" {key} {label} "
            if mode == self._active_screen:
                result.append(segment, style="bold reverse")
            else:
                result.append(segment, style="dim")
            self._regions.append((col, col + len(segment), mode))
            col += len(segment)
            result.append(" ")
            col += 1
        return result

    def on_click(self, event: Click) -> None:
        for start, end, mode in self._regions:
            if start <= event.x < end:
                self.app.switch_mode(mode)
                return
