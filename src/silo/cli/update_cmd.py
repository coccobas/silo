"""CLI: update command — modify a running model server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import typer
from rich.console import Console

from silo.cli.app import app

console = Console()


@app.command("update")
def update(
    name: str = typer.Argument(help="Name of the running model"),
    litellm: bool | None = typer.Option(
        None, "--litellm/--no-litellm", help="Enable/disable LiteLLM registration",
    ),
    litellm_url: str | None = typer.Option(
        None, "--litellm-url", help="LiteLLM proxy URL (e.g. 100.112.188.75)",
    ),
    litellm_api_key: str | None = typer.Option(
        None, "--litellm-api-key", help="LiteLLM API key",
    ),
    litellm_model_name: str | None = typer.Option(
        None, "--litellm-name", help="Model name to register with LiteLLM",
    ),
    model_name: str | None = typer.Option(
        None, "--name", "-n", help="Change the model name",
    ),
    port: int | None = typer.Option(
        None, "--port", "-p", help="Change port (requires restart)",
    ),
) -> None:
    """Update settings on a running model server."""
    from silo.process.pid import read_pid_entry

    entry = read_pid_entry(name)
    if entry is None:
        console.print(f"[red]Model '{name}' is not running.[/red]")
        raise typer.Exit(1)

    server_url = f"http://{entry.host}:{entry.port}"
    changes: list[str] = []

    # Hot changes — talk directly to the model server's admin API
    if litellm is True:
        url = litellm_url
        if not url:
            # Try loading from config
            from silo.config.loader import load_config

            config = load_config()
            url = config.litellm.url

        if not url:
            console.print("[red]--litellm-url required (or set litellm.url in config.yaml)[/red]")
            raise typer.Exit(1)

        from silo.litellm.registry import normalize_litellm_url

        payload: dict[str, str] = {"url": normalize_litellm_url(url)}
        if litellm_api_key:
            payload["api_key"] = litellm_api_key
        else:
            from silo.config.loader import load_config

            config = load_config()
            if config.litellm.api_key:
                payload["api_key"] = config.litellm.api_key
        if litellm_model_name:
            payload["model_name"] = litellm_model_name

        result = _admin_post(server_url, "/admin/litellm/register", payload)
        if result:
            console.print(f"[green]Registered '{name}' with LiteLLM at {payload['url']}[/green]")
            changes.append("litellm_registered")
        else:
            console.print("[red]Failed to register with LiteLLM[/red]")

    elif litellm is False:
        result = _admin_post(server_url, "/admin/litellm/deregister", {})
        if result:
            console.print(f"[green]Deregistered '{name}' from LiteLLM[/green]")
            changes.append("litellm_deregistered")
        else:
            console.print("[red]Failed to deregister from LiteLLM[/red]")

    if model_name:
        result = _admin_put(server_url, "/admin/model-name", {"model_name": model_name})
        if result:
            console.print(f"[green]Renamed to '{model_name}'[/green]")
            changes.append(f"model_name={model_name}")
        else:
            console.print("[red]Failed to rename model[/red]")

    # Cold changes — require restart
    if port and port != entry.port:
        from silo.process.manager import spawn_model, stop_model

        console.print(f"[dim]Restarting {name} on port {port}...[/dim]")
        stop_model(name)
        spawn_model(
            name=name,
            repo_id=entry.repo_id,
            host=entry.host,
            port=port,
        )
        console.print(f"[green]Restarted '{name}' on port {port}[/green]")
        changes.append(f"port={port}")

    if not changes:
        console.print("[dim]No changes requested.[/dim]")


def _admin_post(server_url: str, path: str, data: dict) -> dict | None:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{server_url}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        console.print(f"[dim]Admin call failed: {e}[/dim]")
        return None


def _admin_put(server_url: str, path: str, data: dict) -> dict | None:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{server_url}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        console.print(f"[dim]Admin call failed: {e}[/dim]")
        return None
