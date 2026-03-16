"""Silo CLI — root Typer application."""

from __future__ import annotations

import typer

from silo import __version__

app = typer.Typer(
    name="silo",
    help="A unified CLI for local AI models on Apple Silicon.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"silo {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Silo — your models, locally contained."""


def main() -> None:
    # Deferred imports to register sub-commands
    from silo.cli import (  # noqa: F401
        agent_cmd,
        convert_cmd,
        doctor_cmd,
        down_cmd,
        flow_cmd,
        init_cmd,
        logs_cmd,
        models_cmd,
        ps_cmd,
        run_cmd,
        serve_cmd,
        ui_cmd,
        up_cmd,
        wake_cmd,
    )

    app()
