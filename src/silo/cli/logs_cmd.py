"""CLI: logs command — view model server logs."""

from __future__ import annotations

import typer
from rich.console import Console

from silo.cli.app import app
from silo.config.paths import LOGS_DIR

console = Console()


@app.command("logs")
def logs(
    name: str = typer.Argument(help="Model name to view logs for"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    tail: int = typer.Option(50, "--tail", "-n", help="Number of lines to show"),
) -> None:
    """View logs for a model server."""
    log_file = LOGS_DIR / f"{name}.log"

    if not log_file.exists():
        console.print(f"[yellow]No logs found for '{name}'.[/yellow]")
        raise typer.Exit(1)

    if follow:
        _follow_log(log_file)
    else:
        _tail_log(log_file, tail)


def _tail_log(log_file, n: int) -> None:
    """Show the last n lines of a log file."""
    lines = log_file.read_text().splitlines()
    for line in lines[-n:]:
        console.print(line)


def _follow_log(log_file) -> None:
    """Follow a log file (like tail -f)."""
    import time

    with open(log_file) as f:
        # Seek to end
        f.seek(0, 2)
        try:
            while True:
                line = f.readline()
                if line:
                    console.print(line.rstrip())
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
