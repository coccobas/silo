"""Flows screen — list, inspect, and run flows."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static
from textual import work

from silo.tui.widgets.nav_bar import NavBar


class FlowsScreen(Screen):
    """Flow management: list flows, view steps, run."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("enter", "run_selected", "Run Flow"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(" [b]Flows[/b]", classes="section-title")
        yield DataTable(id="flow-table")
        yield Static(" [b]Steps[/b]", classes="section-title")
        yield DataTable(id="step-table")
        yield Static(id="flow-result", classes="detail-panel")
        yield NavBar(active_screen="flows")
        yield Footer()

    def on_mount(self) -> None:
        flow_t = self.query_one("#flow-table", DataTable)
        flow_t.add_columns("NAME", "DESCRIPTION", "STEPS", "SCHEDULE")
        flow_t.cursor_type = "row"

        step_t = self.query_one("#step-table", DataTable)
        step_t.add_columns("ID", "TYPE", "MODEL", "INPUT", "MAP")
        step_t.cursor_type = "row"

        self.action_refresh()

    def action_refresh(self) -> None:
        self._load_flows()

    @work(thread=True)
    def _load_flows(self) -> None:
        from silo.flows.parser import list_flows

        flows = list_flows()
        rows = [
            (
                flow.name,
                flow.description or "—",
                str(len(flow.steps)),
                flow.schedule or "—",
            )
            for flow in flows
        ]
        self.app.call_from_thread(self._apply_flows, rows)

    def _apply_flows(self, rows) -> None:
        table = self.query_one("#flow-table", DataTable)
        table.clear()
        if rows:
            for name, desc, steps, sched in rows:
                table.add_row(name, desc, steps, sched)
        else:
            table.add_row(
                "[dim]No flows found[/]", "—", "—", "—"
            )

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        if event.data_table.id != "flow-table" or event.row_key is None:
            return
        table = self.query_one("#flow-table", DataTable)
        row_idx = list(table.rows.keys()).index(event.row_key)
        flow_name = str(table.get_cell_at((row_idx, 0)))
        if flow_name.startswith("["):
            return
        self._show_steps(flow_name)

    def _show_steps(self, flow_name: str) -> None:
        from silo.flows.parser import list_flows

        step_table = self.query_one("#step-table", DataTable)
        step_table.clear()

        flows = list_flows()
        for flow in flows:
            if flow.name == flow_name:
                for step in flow.steps:
                    step_table.add_row(
                        step.id,
                        step.type,
                        step.model or "—",
                        step.input or "—",
                        "Yes" if step.map else "No",
                    )
                break

    def action_run_selected(self) -> None:
        table = self.query_one("#flow-table", DataTable)
        if table.cursor_row is None or table.row_count == 0:
            return
        flow_name = str(table.get_cell_at((table.cursor_row, 0)))
        if flow_name.startswith("["):
            return
        self._run_flow(flow_name)

    @work(thread=True)
    def _run_flow(self, flow_name: str) -> None:
        from silo.flows.parser import list_flows
        from silo.flows.runner import run_flow

        flows = list_flows()
        flow_def = next((f for f in flows if f.name == flow_name), None)
        if flow_def is None:
            self.app.call_from_thread(
                self.notify, f"Flow '{flow_name}' not found", severity="error"
            )
            return

        self.app.call_from_thread(
            self.query_one("#flow-result", Static).update,
            f"[yellow]Running '{flow_name}'...[/]",
        )

        try:
            result = run_flow(flow_def)
            lines = [f"[b]{flow_name}[/b] — {'[green]OK[/]' if result.success else '[red]FAILED[/]'}"]
            for sr in result.step_results:
                icon = "[green]✓[/]" if sr.success else "[red]✗[/]"
                lines.append(f"  {icon} {sr.step_id}")
                if sr.error:
                    lines.append(f"    [red]{sr.error}[/]")
            if result.error:
                lines.append(f"[red]{result.error}[/]")

            self.app.call_from_thread(
                self.query_one("#flow-result", Static).update,
                "\n".join(lines),
            )
            severity = "information" if result.success else "error"
            self.app.call_from_thread(
                self.notify,
                f"Flow '{flow_name}' {'completed' if result.success else 'failed'}",
                severity=severity,
            )
        except Exception as exc:
            self.app.call_from_thread(
                self.query_one("#flow-result", Static).update,
                f"[red]Error: {exc}[/]",
            )
            self.app.call_from_thread(
                self.notify, f"Flow error: {exc}", severity="error"
            )
