"""CLI: down command — stop model servers."""

from __future__ import annotations

import typer
from rich.console import Console

from silo.cli.app import app

console = Console()


@app.command("down")
def down(
    name: str | None = typer.Argument(None, help="Model name to stop (all if omitted)"),
) -> None:
    """Stop model server(s)."""
    from silo.process.manager import stop_model
    from silo.process.pid import list_pids

    if name:
        console.print(f"[dim]Stopping {name}...[/dim]")
        stopped = stop_model(name)
        if stopped:
            console.print(f"[green]Stopped {name}[/green]")
        else:
            console.print(f"[yellow]{name} is not running.[/yellow]")
        return

    # Stop all known processes
    pids = list_pids()
    if not pids:
        console.print("[dim]No running models found.[/dim]")
        return

    for model_name in pids:
        console.print(f"[dim]Stopping {model_name}...[/dim]")
        stop_model(model_name)
        console.print(f"[green]Stopped {model_name}[/green]")
