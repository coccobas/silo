"""CLI: doctor command for environment diagnostics."""

from __future__ import annotations

import typer
from rich.console import Console

from silo.cli.app import app

console = Console()


STATUS_ICONS = {
    "ok": "[green]\u2714[/green]",
    "warn": "[yellow]\u26a0[/yellow]",
    "fail": "[red]\u2718[/red]",
}


@app.command("doctor")
def doctor() -> None:
    """Diagnose common setup issues."""
    from silo.doctor.checks import run_all_checks

    console.print("[bold]Silo Doctor[/bold]\n")

    results = run_all_checks()
    has_failures = False

    for result in results:
        icon = STATUS_ICONS.get(result.status.value, "?")
        console.print(f"  {icon} {result.name}: {result.message}")
        if result.status.value == "fail":
            has_failures = True

    console.print()
    if has_failures:
        console.print("[red]Some checks failed. Fix the issues above.[/red]")
        raise typer.Exit(1)

    console.print("[green]All checks passed.[/green]")
