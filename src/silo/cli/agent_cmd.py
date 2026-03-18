"""CLI commands for Silo agent daemon and discovery."""

from __future__ import annotations

import platform

import typer

from silo.cli.app import app

agent_app = typer.Typer(
    name="agent",
    help="Agent daemon management and node discovery.",
    invoke_without_command=True,
)
app.add_typer(agent_app)


@agent_app.callback()
def agent_callback(ctx: typer.Context) -> None:
    """Start the agent daemon (default) or use a subcommand."""
    if ctx.invoked_subcommand is None:
        # Backward compat: `silo agent` without subcommand runs start
        ctx.invoke(start)


@agent_app.command("start")
def start(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address"),
    port: int = typer.Option(9900, "--port", "-p", help="Agent port"),
    name: str = typer.Option(
        None, "--name", "-n", help="Node name for mDNS (default: hostname)"
    ),
    head: bool = typer.Option(
        False, "--head", help="Enable head node mode for cluster coordination"
    ),
) -> None:
    """Start the Silo agent daemon for remote management."""
    import os

    try:
        import uvicorn
    except ImportError:
        typer.echo("uvicorn is required: uv pip install uvicorn", err=True)
        raise typer.Exit(1) from None

    from silo.agent.daemon import create_agent_app
    from silo.config.paths import acquire_agent_lock, read_agent_lock, release_agent_lock, ensure_dirs

    ensure_dirs()

    # Singleton guard — reuse existing agent if already running
    existing = read_agent_lock()
    if existing:
        existing_port = existing["port"]
        typer.echo(
            f"Agent already running (PID {existing['pid']}, port {existing_port}). "
            f"Connecting to existing instance..."
        )
        # Verify it's actually reachable
        import json
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{existing_port}/health", timeout=3
            ) as resp:
                data = json.loads(resp.read())
            typer.echo(
                f"Agent is healthy: {data.get('hostname', '?')} "
                f"v{data.get('version', '?')} on port {existing_port}"
            )
            raise typer.Exit(0)
        except (urllib.error.URLError, OSError):
            typer.echo("Existing agent is not responding. Taking over...")
            release_agent_lock()

    if not acquire_agent_lock(os.getpid(), port):
        typer.echo("Could not acquire agent lock.", err=True)
        raise typer.Exit(1)

    node_name = name or platform.node()
    agent_app_instance = create_agent_app(
        node_name=node_name, port=port, head=head
    )
    mode = "head" if head else "worker"
    typer.echo(f"Silo agent '{node_name}' ({mode}) listening on {host}:{port}")
    try:
        uvicorn.run(agent_app_instance, host=host, port=port)
    finally:
        release_agent_lock()


@agent_app.command("discover")
def discover(
    timeout: float = typer.Option(
        3.0, "--timeout", "-t", help="Discovery timeout in seconds"
    ),
) -> None:
    """Discover Silo agent nodes on the local network via mDNS."""
    try:
        from silo.agent.discovery import discover_nodes
    except ImportError:
        typer.echo(
            "zeroconf is required for discovery.\n"
            "Install with: pip install silo[discovery]",
            err=True,
        )
        raise typer.Exit(1) from None

    from rich.console import Console
    from rich.table import Table

    typer.echo(f"Scanning for Silo agents ({timeout}s)...")
    nodes = discover_nodes(timeout=timeout)

    if not nodes:
        typer.echo("No agents found on the network.")
        raise typer.Exit(0)

    table = Table(title="Discovered Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Host", style="green")
    table.add_column("Port", style="yellow")
    table.add_column("Hostname", style="dim")

    for node in nodes:
        table.add_row(node.name, node.host, str(node.port), node.hostname)

    Console().print(table)
    typer.echo(f"\nFound {len(nodes)} agent(s).")
