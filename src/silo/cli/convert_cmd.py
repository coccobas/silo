"""CLI: convert command."""

from __future__ import annotations

import typer
from rich.console import Console

from silo.cli.app import app

console = Console()


@app.command("convert")
def convert(
    repo_id: str = typer.Argument(help="HuggingFace repository ID to convert"),
    quantize: str | None = typer.Option(
        None, "--quantize", "-q", help="Quantization type (q4, q8)"
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """Convert a model to MLX format."""
    from silo.convert.mlx import convert_model
    from silo.registry.models import ModelFormat, RegistryEntry
    from silo.registry.store import Registry

    console.print(f"[bold]Converting {repo_id}...[/bold]")
    if quantize:
        console.print(f"  Quantization: {quantize}")

    try:
        with console.status("Converting..."):
            output_path = convert_model(repo_id, quantize=quantize, output=output)
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Conversion failed: {e}[/red]")
        raise typer.Exit(1)

    entry = RegistryEntry(
        repo_id=repo_id,
        format=ModelFormat.MLX,
        local_path=str(output_path),
        tags=["converted"] + ([f"quantized-{quantize}"] if quantize else []),
    )
    registry = Registry.load().add(entry)
    registry.save()

    console.print(f"[green]Converted to {output_path}[/green]")
    console.print("[dim]Registered in local registry.[/dim]")
