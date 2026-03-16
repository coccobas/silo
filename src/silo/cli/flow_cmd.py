"""CLI: flow command — manage and run flows."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from silo.cli.app import app

console = Console()

flow_app = typer.Typer(name="flow", help="Manage and run workflow pipelines.")
app.add_typer(flow_app)


@flow_app.command("list")
def flow_list() -> None:
    """List available flow definitions."""

    from silo.config.paths import CONFIG_DIR
    from silo.flows.parser import list_flows

    flows_dir = CONFIG_DIR / "flows"
    flows = list_flows(flows_dir)

    if not flows:
        console.print("[dim]No flows found. Create YAML flow files in ~/.silo/flows/[/dim]")
        return

    table = Table(title="Available Flows")
    table.add_column("NAME", style="bold")
    table.add_column("DESCRIPTION")
    table.add_column("STEPS")
    table.add_column("SCHEDULE")

    for flow in flows:
        table.add_row(
            flow.name,
            flow.description or "—",
            str(len(flow.steps)),
            flow.schedule or "—",
        )

    console.print(table)


@flow_app.command("run")
def flow_run(
    name: str = typer.Argument(help="Flow name or path to YAML file"),
    input_data: str | None = typer.Option(None, "--input", "-i", help="Input data or file path"),
) -> None:
    """Run a flow definition."""
    from pathlib import Path

    from silo.config.paths import CONFIG_DIR
    from silo.flows.parser import parse_flow
    from silo.flows.runner import run_flow

    # Try as file path first, then look in flows directory
    flow_path = Path(name)
    if not flow_path.exists():
        flow_path = CONFIG_DIR / "flows" / f"{name}.yaml"

    if not flow_path.exists():
        console.print(f"[red]Flow '{name}' not found.[/red]")
        raise typer.Exit(1)

    try:
        flow = parse_flow(flow_path)
    except ValueError as e:
        console.print(f"[red]Invalid flow: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Running flow: {flow.name}[/green]")
    console.print(f"[dim]Steps: {len(flow.steps)}[/dim]")

    result = run_flow(flow, input_data=input_data)

    if result.success:
        console.print("[green]Flow completed successfully.[/green]")
        if result.final_output:
            console.print(f"Output: {result.final_output}")
    else:
        console.print(f"[red]Flow failed: {result.error}[/red]")
        raise typer.Exit(1)
