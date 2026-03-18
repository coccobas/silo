"""CLI: serve command — start an OpenAI-compatible model server."""

from __future__ import annotations

import os

import typer
from rich.console import Console

from silo.cli.app import app

console = Console()


@app.command("serve")
def serve(
    repo_id: str = typer.Argument(help="Model repository ID"),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Bind host"),
    port: int = typer.Option(8800, "--port", "-p", help="Bind port"),
    name: str | None = typer.Option(None, "--name", "-n", help="Friendly model name"),
    quantize: str | None = typer.Option(
        None, "--quantize", "-q", help="Quantize during auto-convert"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output path for auto-convert"
    ),
) -> None:
    """Serve a model behind an OpenAI-compatible API."""
    import uvicorn

    from silo.backends.factory import resolve_backend
    from silo.download.hf import download_model, get_model_info
    from silo.registry.detector import detect_model_format
    from silo.registry.models import ModelFormat, RegistryEntry
    from silo.registry.store import Registry
    from silo.server.app import create_app

    # Env var overrides
    host = os.environ.get("SILO_HOST", host)
    port = int(os.environ.get("SILO_PORT", str(port)))

    model_name = name or repo_id

    # Resolve model
    registry = Registry.load()
    entry = registry.get(repo_id)

    if not entry:
        console.print(f"[dim]Resolving {repo_id}...[/dim]")
        try:
            info = get_model_info(repo_id)
            fmt = detect_model_format(repo_id, info.get("siblings", []))
        except Exception as e:
            console.print(f"[red]Could not fetch model info: {e}[/red]")
            raise typer.Exit(1)

        if fmt == ModelFormat.STANDARD and quantize:
            console.print(f"[dim]Converting {repo_id} with quantize={quantize}...[/dim]")
            from silo.convert.mlx import convert_model

            try:
                local_path = convert_model(repo_id, quantize=quantize, output=output)
                fmt = ModelFormat.MLX
            except ImportError as e:
                console.print(f"[red]{e}[/red]")
                raise typer.Exit(1)
        elif fmt == ModelFormat.STANDARD:
            console.print(
                f"[yellow]Model is standard format. Use --quantize to auto-convert,[/yellow]\n"
                f"[yellow]or convert first: silo convert {repo_id} --quantize q4[/yellow]"
            )
            raise typer.Exit(1)
        else:
            with console.status(f"Downloading {repo_id}..."):
                local_path = download_model(repo_id)

        entry = RegistryEntry(
            repo_id=repo_id,
            format=fmt,
            local_path=str(local_path),
        )
        registry = registry.add(entry)
        registry.save()

    model_path = entry.local_path or repo_id

    # Load backend
    try:
        backend = resolve_backend(entry.format)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Loading {repo_id}...[/dim]")
    try:
        backend.load(model_path, {})
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Create and run server
    server_app = create_app(backend, model_name)

    # Show appropriate endpoints based on model type
    endpoints = ["GET  /v1/models", "GET  /health"]
    if entry.format in (ModelFormat.MLX, ModelFormat.STANDARD, ModelFormat.GGUF):
        endpoints.insert(0, "POST /v1/chat/completions")
    if entry.format == ModelFormat.AUDIO_STT:
        endpoints.insert(0, "POST /v1/audio/transcriptions")
    if entry.format == ModelFormat.AUDIO_TTS:
        endpoints.insert(0, "POST /v1/audio/speech")

    endpoint_lines = "\n".join(f"  {e}" for e in endpoints)
    console.print(
        f"[green]Serving {model_name} on http://{host}:{port}[/green]\n"
        f"[dim]{endpoint_lines}[/dim]"
    )

    # Register with LiteLLM if configured
    import uuid

    from silo.config.loader import load_config
    from silo.litellm.registry import deregister_model, register_model

    config = load_config()
    instance_id = str(uuid.uuid4())
    register_model(config.litellm, model_name, host, port, instance_id)

    try:
        uvicorn.run(server_app, host=host, port=port, log_level="warning")
    finally:
        deregister_model(config.litellm, model_name, instance_id)
