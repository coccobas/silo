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
    from silo.config.loader import load_config
    from silo.litellm.registry import deregister_model
    from silo.process.manager import stop_model
    from silo.process.pid import list_pid_entries, read_pid_entry

    config = load_config()

    if name:
        # Read instance_id before stopping so we can deregister
        entry = read_pid_entry(name)
        console.print(f"[dim]Stopping {name}...[/dim]")
        stopped = stop_model(name)
        if stopped:
            console.print(f"[green]Stopped {name}[/green]")
            if entry:
                deregister_model(config.litellm, name, entry.instance_id)
        else:
            console.print(f"[yellow]{name} is not running.[/yellow]")
        return

    # Stop all known processes
    entries = list_pid_entries()
    if not entries:
        console.print("[dim]No running models found.[/dim]")
        return

    for model_name, entry in entries.items():
        console.print(f"[dim]Stopping {model_name}...[/dim]")
        stop_model(model_name)
        console.print(f"[green]Stopped {model_name}[/green]")
        deregister_model(config.litellm, model_name, entry.instance_id)
