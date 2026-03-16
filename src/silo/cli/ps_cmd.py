"""CLI: ps command — show running model servers."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from silo.cli.app import app

console = Console()


@app.command("ps")
def ps() -> None:
    """Show status of running model servers."""
    from silo.config.loader import load_config
    from silo.process.manager import get_status

    config = load_config()

    if not config.models:
        # Fall back to listing from PID files
        from silo.process.manager import list_running

        running = list_running()
        if not running:
            console.print("[dim]No running models found.[/dim]")
            return

        table = Table(title="Model Servers")
        table.add_column("NAME", style="bold")
        table.add_column("PID")
        table.add_column("STATUS")

        for proc in running:
            status_style = "green" if proc.status == "running" else "red"
            table.add_row(proc.name, str(proc.pid), f"[{status_style}]{proc.status}[/]")

        console.print(table)
        return

    table = Table(title="Model Servers")
    table.add_column("NAME", style="bold")
    table.add_column("REPO")
    table.add_column("PORT")
    table.add_column("PID")
    table.add_column("STATUS")

    for model in config.models:
        proc = get_status(model.name, port=model.port, repo_id=model.repo)
        status_style = "green" if proc.status == "running" else "red"
        pid_str = str(proc.pid) if proc.pid else "—"
        table.add_row(
            proc.name,
            model.repo,
            str(model.port),
            pid_str,
            f"[{status_style}]{proc.status}[/]",
        )

    console.print(table)
