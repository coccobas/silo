"""CLI: ui command — launch the TUI dashboard."""

from __future__ import annotations

import typer
from rich.console import Console

from silo.cli.app import app

console = Console()


@app.command("ui")
def ui() -> None:
    """Launch the TUI dashboard (requires silo[tui])."""
    try:
        from silo.tui.app import create_tui_app
    except ImportError:
        console.print(
            "[red]Textual is required for the TUI.[/red]\n"
            "[dim]Install with: uv pip install 'silo[tui]'[/dim]"
        )
        raise typer.Exit(1)

    tui_app = create_tui_app()
    tui_app.run()
