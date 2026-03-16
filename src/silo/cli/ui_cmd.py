"""CLI: ui command — launch the TUI dashboard."""

from __future__ import annotations

import typer
from rich.console import Console

from silo.cli.app import app

console = Console()


@app.command("ui")
def ui(
    head: bool = typer.Option(
        False, "--head", help="Also start the agent daemon as head node."
    ),
    worker: bool = typer.Option(
        False, "--worker", help="Also start the agent daemon as a worker node."
    ),
    agent_host: str = typer.Option(
        "0.0.0.0", "--host", help="Agent daemon bind address (used with --head/--worker)."
    ),
    agent_port: int = typer.Option(
        9900, "--agent-port", help="Agent daemon port (used with --head/--worker)."
    ),
    name: str | None = typer.Option(
        None, "--name", "-n", help="Node name (default: hostname)."
    ),
) -> None:
    """Launch the TUI dashboard (requires silo[tui])."""
    if head and worker:
        console.print("[red]Cannot use --head and --worker together.[/red]")
        raise typer.Exit(1)

    try:
        from silo.tui.app import create_tui_app
    except ImportError:
        console.print(
            "[red]Textual is required for the TUI.[/red]\n"
            "[dim]Install with: uv pip install 'silo[tui]'[/dim]"
        )
        raise typer.Exit(1)

    if head or worker:
        _start_agent_daemon(
            agent_host=agent_host,
            agent_port=agent_port,
            name=name,
            head=head,
        )

    tui_app = create_tui_app()
    if head:
        tui_app.agent_head_port = agent_port
    tui_app.run()


def _start_agent_daemon(
    *,
    agent_host: str,
    agent_port: int,
    name: str | None,
    head: bool,
) -> None:
    """Start the agent daemon in a background thread."""
    import platform as plat
    import socket
    import threading
    import time

    import uvicorn

    from silo.agent.daemon import create_agent_app
    from silo.config.paths import ensure_dirs

    node_name = name or plat.node()
    ensure_dirs()
    agent_instance = create_agent_app(
        node_name=node_name, port=agent_port, head=head
    )
    server = uvicorn.Server(uvicorn.Config(
        agent_instance, host=agent_host, port=agent_port, log_level="warning"
    ))
    mode = "head" if head else "worker"
    thread = threading.Thread(
        target=server.run, daemon=True, name=f"agent-{mode}"
    )
    thread.start()

    # Wait for the agent to be ready before launching TUI
    for _ in range(50):
        try:
            s = socket.create_connection(
                ("127.0.0.1", agent_port), timeout=0.5
            )
            s.close()
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)

    console.print(
        f"[dim]Agent {mode} '{node_name}' started on {agent_host}:{agent_port}[/dim]"
    )
