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
    elif worker:
        # Discover the head node via mDNS so the cluster screen works
        head_url = _discover_head()
        if head_url:
            tui_app.cluster_head_url = head_url
            console.print(f"[dim]Discovered head node at {head_url}[/dim]")
    tui_app.run()


def _discover_head() -> str | None:
    """Try to find a head node on the network via mDNS."""
    try:
        from silo.agent.discovery import discover_nodes

        nodes = discover_nodes(timeout=2.0)
        for node in nodes:
            if node.role == "head":
                return f"http://{node.host}:{node.port}"
    except Exception:
        pass
    return None


def _start_agent_daemon(
    *,
    agent_host: str,
    agent_port: int,
    name: str | None,
    head: bool,
) -> None:
    """Start the agent daemon in a background thread."""
    import platform as plat
    import threading
    import time

    import uvicorn

    from silo.agent.daemon import create_agent_app
    from silo.config.paths import ensure_dirs

    import socket
    import urllib.error
    import urllib.request

    mode = "head" if head else "worker"

    # Check if port is already in use
    try:
        s = socket.create_connection(("127.0.0.1", agent_port), timeout=0.5)
        s.close()
        console.print(
            f"[red]Port {agent_port} is already in use. "
            f"Kill the existing process or use --agent-port.[/red]"
        )
        raise typer.Exit(1)
    except (ConnectionRefusedError, OSError):
        pass  # Port is free — good

    node_name = name or plat.node()
    ensure_dirs()
    agent_instance = create_agent_app(
        node_name=node_name, port=agent_port, head=head
    )
    server = uvicorn.Server(uvicorn.Config(
        agent_instance, host=agent_host, port=agent_port, log_level="warning"
    ))
    startup_error: list[BaseException] = []

    def _run_server() -> None:
        try:
            server.run()
        except BaseException as exc:
            startup_error.append(exc)

    thread = threading.Thread(
        target=_run_server, daemon=True, name=f"agent-{mode}"
    )
    thread.start()

    # Wait for the agent to be fully ready (lifespan complete)
    check_path = "/health"
    ready = False
    for _ in range(100):
        if startup_error:
            console.print(
                f"[red]Agent {mode} failed to start: {startup_error[0]}[/red]"
            )
            raise typer.Exit(1)
        if not thread.is_alive():
            console.print(
                f"[red]Agent {mode} exited unexpectedly[/red]"
            )
            raise typer.Exit(1)
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{agent_port}{check_path}", timeout=1
            )
            ready = True
            break
        except Exception:
            time.sleep(0.1)

    if not ready:
        console.print(
            f"[red]Agent {mode} did not become ready within 10s[/red]"
        )
        raise typer.Exit(1)

    console.print(
        f"[dim]Agent {mode} '{node_name}' started on {agent_host}:{agent_port}[/dim]"
    )
