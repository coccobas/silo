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
    agent_port: int = typer.Option(
        9900, "--agent-port", help="Agent daemon port (used with --head)."
    ),
    name: str | None = typer.Option(
        None, "--name", "-n", help="Node name (default: hostname)."
    ),
) -> None:
    """Launch the TUI dashboard (requires silo[tui])."""
    try:
        from silo.tui.app import create_tui_app
    except ImportError:
        console.print(
            "[red]Textual is required for the TUI.[/red]\n"
            "[dim]Install with: uv pip install 'silo[tui]'[/dim]"
        )
        raise typer.Exit(1)

    agent_thread = None
    if head:
        import platform as plat
        import threading
        import time

        import uvicorn

        from silo.agent.daemon import create_agent_app
        from silo.config.paths import ensure_dirs

        node_name = name or plat.node()
        ensure_dirs()
        agent_instance = create_agent_app(
            node_name=node_name, port=agent_port, head=True
        )
        server = uvicorn.Server(uvicorn.Config(
            agent_instance, host="0.0.0.0", port=agent_port, log_level="warning"
        ))
        agent_thread = threading.Thread(
            target=server.run, daemon=True, name="agent-head"
        )
        agent_thread.start()

        # Wait for the agent to be ready before launching TUI
        import socket

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
            f"[dim]Agent head '{node_name}' started on :{agent_port}[/dim]"
        )

    tui_app = create_tui_app()
    if head:
        tui_app.agent_head_port = agent_port
    tui_app.run()
