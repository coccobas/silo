"""Doctor screen — environment diagnostic checks."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static
from textual import work

from silo.tui.widgets.nav_bar import NavBar


class DoctorScreen(Screen):
    """Run and display environment diagnostic checks."""

    BINDINGS = [
        ("r", "refresh", "Re-run Checks"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(" [b]Doctor — Environment Checks[/b]", classes="section-title")
        yield DataTable(id="checks-table")
        yield NavBar(active_screen="doctor")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#checks-table", DataTable)
        table.add_columns("CHECK", "STATUS", "DETAIL")
        table.cursor_type = "row"
        self.action_refresh()

    def action_refresh(self) -> None:
        self._run_checks()

    @work(thread=True)
    def _run_checks(self) -> None:
        from silo.doctor.checks import run_all_checks

        checks = run_all_checks()
        rows = []
        for check in checks:
            rows.append((check.name, check.status, check.message))
        self.app.call_from_thread(self._apply_checks, rows)

    def _apply_checks(self, rows) -> None:
        table = self.query_one("#checks-table", DataTable)
        table.clear()
        color_map = {"ok": "green", "warn": "yellow", "fail": "red"}
        for name, status, message in rows:
            c = color_map.get(status, "dim")
            table.add_row(
                name,
                f"[{c}]{status}[/]",
                message,
            )
