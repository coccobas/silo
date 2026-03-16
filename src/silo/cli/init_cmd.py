"""CLI: init command to generate starter config."""

from __future__ import annotations

import typer
from rich.console import Console

from silo.cli.app import app
from silo.config.paths import CONFIG_FILE, ensure_dirs

console = Console()

STARTER_CONFIG = """\
# Silo configuration
# See: https://github.com/silo/silo for documentation

models:
  # Example: LLM model
  # - name: llama
  #   repo: mlx-community/Llama-3.2-1B-4bit
  #   host: 127.0.0.1
  #   port: 8800
  #   warmup: false
  #   restart: on-failure
  #   timeout: 120

  # Example: STT model
  # - name: whisper
  #   repo: mlx-community/whisper-large-v3-turbo
  #   host: 0.0.0.0
  #   port: 8801
  #   warmup: true
  #   restart: always
  #   timeout: 60

  # Example: TTS model
  # - name: tts
  #   repo: mlx-community/kokoro-tts
  #   host: 0.0.0.0
  #   port: 8802
  #   restart: always
  #   timeout: 30
"""


@app.command("init")
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
) -> None:
    """Generate a starter config.yaml."""
    ensure_dirs()

    if CONFIG_FILE.exists() and not force:
        console.print(f"[yellow]Config already exists at {CONFIG_FILE}[/yellow]")
        console.print("[dim]Use --force to overwrite.[/dim]")
        raise typer.Exit(1)

    CONFIG_FILE.write_text(STARTER_CONFIG)
    console.print(f"[green]Created config at {CONFIG_FILE}[/green]")
    console.print(
        "[dim]Edit the file to add your models, then run 'silo doctor' to verify.[/dim]"
    )
