"""CLI: run command for one-shot inference."""

from __future__ import annotations

import typer
from rich.console import Console

from silo.cli.app import app

console = Console()


@app.command("run")
def run(
    repo_id: str = typer.Argument(help="Model repository ID"),
    prompt: str = typer.Argument(help="Prompt text"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-m", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream output"),
) -> None:
    """Run one-shot inference with a model."""
    from silo.backends.mlx_lm import MlxLmBackend
    from silo.download.hf import download_model, get_model_info
    from silo.registry.detector import detect_model_format
    from silo.registry.models import ModelFormat, RegistryEntry
    from silo.registry.store import Registry

    registry = Registry.load()
    entry = registry.get(repo_id)

    # Auto-download if not registered
    if not entry:
        console.print("[dim]Model not in registry. Checking HuggingFace...[/dim]")
        try:
            info = get_model_info(repo_id)
            fmt = detect_model_format(repo_id, info.get("siblings", []))
        except Exception as e:
            console.print(f"[red]Could not fetch model info: {e}[/red]")
            raise typer.Exit(1)

        if fmt == ModelFormat.STANDARD:
            console.print(
                f"[yellow]Model is in standard format. Convert first with:[/yellow]\n"
                f"  silo convert {repo_id} --quantize q4"
            )
            raise typer.Exit(1)

        with console.status(f"Downloading {repo_id}..."):
            local_path = download_model(repo_id)

        entry = RegistryEntry(
            repo_id=repo_id,
            format=fmt,
            local_path=str(local_path),
        )
        registry = registry.add(entry)
        registry.save()
        console.print("[green]Downloaded and registered.[/green]")

    model_path = entry.local_path or repo_id

    # Load and run
    backend = MlxLmBackend()
    try:
        with console.status("Loading model..."):
            backend.load(model_path)
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    messages = [{"role": "user", "content": prompt}]

    try:
        if stream:
            result = backend.chat(
                messages, stream=True, max_tokens=max_tokens, temperature=temperature
            )
            for chunk in result:
                text = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if text:
                    typer.echo(text, nl=False)
            typer.echo()
        else:
            result = backend.chat(
                messages, stream=False, max_tokens=max_tokens, temperature=temperature
            )
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            typer.echo(text)
    finally:
        backend.unload()
