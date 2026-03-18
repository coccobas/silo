"""CLI: up command — bring up models from config."""

from __future__ import annotations

import typer
from rich.console import Console

from silo.cli.app import app

console = Console()


@app.command("up")
def up(
    name: str | None = typer.Argument(None, help="Model name to bring up (all if omitted)"),
) -> None:
    """Bring up model server(s) from config.yaml."""
    from silo.config.loader import load_config
    from silo.process.manager import spawn_model
    from silo.process.pid import is_running, read_pid

    config = load_config()
    if not config.models:
        console.print("[yellow]No models configured. Run 'silo init' first.[/yellow]")
        raise typer.Exit(1)

    models = config.models
    if name:
        models = [m for m in models if m.name == name]
        if not models:
            console.print(f"[red]Model '{name}' not found in config.[/red]")
            raise typer.Exit(1)

    for model in models:
        existing_pid = read_pid(model.name)
        if existing_pid and is_running(existing_pid):
            console.print(f"[dim]{model.name} already running (PID {existing_pid})[/dim]")
            continue

        console.print(
            f"[green]Starting {model.name} ({model.repo}) on port {model.port}...[/green]"
        )
        result = spawn_model(
            name=model.name,
            repo_id=model.repo,
            host=model.host,
            port=model.port,
            quantize=model.quantize,
            output=model.output,
        )
        console.print(f"[dim]  PID {result.pid}[/dim]")

        # Register with LiteLLM if enabled
        from silo.litellm.registry import register_model

        register_model(
            config.litellm, model.name, model.host, model.port, result.instance_id,
        )
