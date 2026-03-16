"""CLI: wake command — listen for a wake word and trigger a flow."""

from __future__ import annotations

import signal

import typer
from rich.console import Console

from silo.cli.app import app

console = Console()


@app.command("wake")
def wake(
    word: str = typer.Option(
        "hey_jarvis",
        "--word",
        "-w",
        help="Wake word name (built-in: hey_jarvis, alexa, etc.) or path to .onnx model.",
    ),
    flow: str = typer.Option(
        ...,
        "--flow",
        "-f",
        help="Flow name or path to YAML flow file.",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="Detection confidence threshold (0.0-1.0).",
    ),
    continuous: bool = typer.Option(
        True,
        "--continuous/--once",
        help="Keep listening after flow completes, or stop after first detection.",
    ),
    device: int | None = typer.Option(
        None,
        "--device",
        "-d",
        help="Audio device index (default: system default mic).",
    ),
) -> None:
    """Listen for a wake word and trigger a flow on detection."""
    from pathlib import Path

    from silo.config.paths import CONFIG_DIR
    from silo.flows.parser import parse_flow
    from silo.flows.runner import run_flow

    # Validate flow exists
    flow_path = Path(flow)
    if not flow_path.exists():
        flow_path = CONFIG_DIR / "flows" / f"{flow}.yaml"
    if not flow_path.exists():
        console.print(f"[red]Flow '{flow}' not found.[/red]")
        console.print(f"[dim]Looked in: {flow_path}[/dim]")
        raise typer.Exit(1)

    try:
        flow_def = parse_flow(flow_path)
    except ValueError as e:
        console.print(f"[red]Invalid flow: {e}[/red]")
        raise typer.Exit(1)

    # Resolve model path for custom .onnx files
    model_path = None
    if word.endswith(".onnx") or "/" in word:
        model_path = word

    def run_the_flow(flow_name: str) -> None:
        result = run_flow(flow_def)
        if result.success:
            console.print(f"[green]Flow '{flow_def.name}' completed.[/green]")
            if result.final_output:
                console.print(f"[dim]Output: {result.final_output}[/dim]")
        else:
            console.print(f"[red]Flow failed: {result.error}[/red]")

    def on_status(status) -> None:
        from silo.wake.listener import WakeState

        if status.state == WakeState.LISTENING:
            mode = "continuous" if continuous else "once"
            console.print(
                f"[green]Listening for '{word}'...[/green] "
                f"[dim]({mode} mode, flow: {flow_def.name})[/dim]"
            )
        elif status.state == WakeState.DETECTED:
            console.print(
                f"[bold yellow]Wake word detected![/bold yellow] "
                f"[dim](detection #{status.detections})[/dim]"
            )
        elif status.state == WakeState.RUNNING_FLOW:
            console.print(f"[cyan]Running flow '{flow_def.name}'...[/cyan]")
        elif status.state == WakeState.ERROR:
            console.print(f"[red]Error: {status.error}[/red]")
        elif status.state == WakeState.STOPPED:
            console.print("[dim]Stopped.[/dim]")

    # Build listener
    from silo.wake.listener import ListenerConfig, WakeWordListener

    listener_config = ListenerConfig(
        wake_word=word,
        flow_name=flow_def.name,
        threshold=threshold,
        model_path=model_path,
        continuous=continuous,
        device=device,
    )
    listener = WakeWordListener(
        config=listener_config,
        flow_runner=run_the_flow,
        on_status=on_status,
    )

    # Handle Ctrl+C gracefully
    def handle_sigint(signum: int, frame: object) -> None:
        listener.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        listener.run()
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
