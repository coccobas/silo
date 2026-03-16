"""CLI: models list|info|rm|search commands."""

from __future__ import annotations

import shutil

import typer
from rich.console import Console
from rich.table import Table

from silo.cli.app import app

models_app = typer.Typer(name="models", help="Manage local models.", no_args_is_help=True)
app.add_typer(models_app)

console = Console()


@models_app.command("list")
def models_list() -> None:
    """List all locally registered models."""
    from silo.registry.store import Registry

    registry = Registry.load()
    entries = registry.list()

    if not entries:
        console.print(
            "[dim]No models registered. Use 'silo models search' to find models.[/dim]"
        )
        return

    table = Table(title="Local Models")
    table.add_column("Repository", style="cyan")
    table.add_column("Format", style="green")
    table.add_column("Path", style="dim")
    table.add_column("Downloaded", style="dim")

    for entry in entries:
        table.add_row(
            entry.repo_id,
            entry.format.value,
            entry.local_path or "—",
            entry.downloaded_at[:10] if entry.downloaded_at else "—",
        )

    console.print(table)


@models_app.command("info")
def models_info(repo_id: str = typer.Argument(help="Model repository ID")) -> None:
    """Show detailed info for a model."""
    from silo.registry.store import Registry

    registry = Registry.load()
    entry = registry.get(repo_id)

    if entry:
        console.print(f"[bold cyan]{entry.repo_id}[/bold cyan]")
        console.print(f"  Format:      {entry.format.value}")
        console.print(f"  Local path:  {entry.local_path or '—'}")
        console.print(f"  Downloaded:  {entry.downloaded_at}")
        if entry.size_bytes:
            size_mb = entry.size_bytes / (1024 * 1024)
            console.print(f"  Size:        {size_mb:.1f} MB")
        if entry.tags:
            console.print(f"  Tags:        {', '.join(entry.tags)}")
    else:
        console.print("[dim]Not in local registry. Fetching from HuggingFace...[/dim]")
        try:
            from silo.download.hf import get_model_info

            info = get_model_info(repo_id)
            console.print(f"[bold cyan]{info['id']}[/bold cyan]")
            console.print(f"  Author:      {info.get('author', '—')}")
            console.print(f"  Downloads:   {info.get('downloads', '—')}")
            console.print(f"  Likes:       {info.get('likes', '—')}")
            console.print(f"  Pipeline:    {info.get('pipeline_tag', '—')}")
            console.print(f"  Library:     {info.get('library_name', '—')}")
        except Exception as e:
            console.print(f"[red]Error fetching model info: {e}[/red]")
            raise typer.Exit(1)


@models_app.command("rm")
def models_rm(
    repo_id: str = typer.Argument(help="Model repository ID"),
    purge: bool = typer.Option(False, "--purge", help="Also delete cached model files"),
) -> None:
    """Remove a model from the registry."""
    from silo.registry.store import Registry

    registry = Registry.load()
    entry = registry.get(repo_id)

    if not entry:
        console.print(f"[yellow]Model '{repo_id}' not in registry.[/yellow]")
        raise typer.Exit(1)

    if purge and entry.local_path:
        path = entry.local_path
        if typer.confirm(f"Delete files at {path}?"):
            shutil.rmtree(path, ignore_errors=True)
            console.print(f"[dim]Deleted {path}[/dim]")

    updated = registry.remove(repo_id)
    updated.save()
    console.print(f"[green]Removed '{repo_id}' from registry.[/green]")


@models_app.command("search")
def models_search(
    query: str = typer.Argument(help="Search query"),
    mlx_only: bool = typer.Option(False, "--mlx-only", help="Show only MLX models"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
) -> None:
    """Search HuggingFace Hub for models."""
    from silo.download.hf import search_models

    with console.status("Searching HuggingFace Hub..."):
        results = search_models(query, mlx_only=mlx_only, limit=limit)

    if not results:
        console.print("[dim]No models found.[/dim]")
        return

    table = Table(title=f"Search: {query}")
    table.add_column("Repository", style="cyan")
    table.add_column("Downloads", justify="right")
    table.add_column("Likes", justify="right")
    table.add_column("Pipeline", style="dim")

    for m in results:
        table.add_row(
            m["id"],
            str(m.get("downloads", "—")),
            str(m.get("likes", "—")),
            m.get("pipeline_tag") or "—",
        )

    console.print(table)
