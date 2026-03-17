"""CLI: ps command — show running model servers."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from silo.cli.app import app

console = Console()


def _print_node_stats() -> None:
    """Print a compact node stats summary (CPU, GPU, memory)."""
    from silo.agent.client import build_clients
    from silo.config.loader import load_config

    config = load_config()
    clients = build_clients(config.nodes)

    table = Table(title="Nodes")
    table.add_column("NODE", style="bold")
    table.add_column("STATUS")
    table.add_column("CPU")
    table.add_column("GPU")
    table.add_column("MEMORY")

    for name, client in clients.items():
        try:
            mem = client.memory()
            mem_str = f"{mem.usage_percent:.0f}% of {mem.total_gb:.0f} GB"
            pressure_color = {
                "normal": "green",
                "warn": "yellow",
                "critical": "red",
            }.get(mem.pressure, "")

            try:
                stats = client.system_stats()
                cpu_str = f"{stats.cpu_percent:.0f}%"
                gpu_str = (
                    f"{stats.gpu_percent:.0f}%"
                    if stats.gpu_percent > 0
                    else "—"
                )
            except Exception:
                cpu_str = "—"
                gpu_str = "—"

            table.add_row(
                name,
                "[green]online[/]",
                cpu_str,
                gpu_str,
                f"[{pressure_color}]{mem_str}[/]" if pressure_color else mem_str,
            )
        except Exception:
            table.add_row(name, "[red]offline[/]", "—", "—", "—")

    console.print(table)


@app.command("ps")
def ps() -> None:
    """Show status of running model servers."""
    from silo.config.loader import load_config
    from silo.process.manager import get_status

    config = load_config()

    _print_node_stats()
    console.print()

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
