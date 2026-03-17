"""Flow execution engine — runs flow definitions."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from silo.flows.parser import FlowDefinition, FlowStep


@dataclass
class StepResult:
    """Result of executing a single flow step."""

    step_id: str
    success: bool
    output: Any = None
    error: str | None = None


@dataclass
class FlowResult:
    """Result of executing a complete flow."""

    flow_name: str
    success: bool
    step_results: list[StepResult] = field(default_factory=list)
    final_output: Any = None
    error: str | None = None


def _find_model_endpoint(
    model_name: str, node_hint: str | None = None
) -> tuple[str, int]:
    """Find the host and port for a model by looking up config and running processes.

    Searches config models by name, then by repo_id. For remote nodes,
    uses the node's host address instead of the model's bind address.

    Args:
        model_name: Model name or repo ID to look up.
        node_hint: Optional node name to constrain the search to.

    Returns:
        (host, port) tuple for the running model server.

    Raises:
        ValueError: If the model is not found or not running.
    """
    from silo.agent.client import build_clients, local_node_name
    from silo.config.loader import load_config

    config = load_config()
    clients = build_clients(config.nodes)
    local_name = local_node_name()

    # Try to match by model name first, then by repo_id
    for model_cfg in config.models:
        if model_cfg.name == model_name or model_cfg.repo == model_name:
            node_name = model_cfg.node or local_name

            # Skip if a node hint is given and this model is on a different node
            if node_hint and node_name != node_hint:
                continue

            client = clients.get(node_name)
            if client is None:
                raise ValueError(
                    f"Node '{node_name}' not found for model '{model_name}'"
                )

            status = client.get_status(
                model_cfg.name,
                port=model_cfg.port,
                repo_id=model_cfg.repo,
            )
            if status.status != "running":
                raise ValueError(
                    f"Model '{model_name}' is not running on {node_name}. "
                    f"Start it with: silo up"
                )

            # For remote nodes, use the node's host, not the model's bind address
            if node_name == local_name:
                host = model_cfg.host
            else:
                node_cfg = next(
                    (n for n in config.nodes if n.name == node_name), None
                )
                host = node_cfg.host if node_cfg else model_cfg.host

            return host, model_cfg.port

    # Not in config — scan all nodes for a running process with this name
    nodes_to_scan = (
        {node_hint: clients[node_hint]}
        if node_hint and node_hint in clients
        else clients
    )
    for node_name, client in nodes_to_scan.items():
        try:
            processes = client.list_processes()
        except Exception:
            continue
        for proc in processes:
            if proc.name == model_name and proc.status == "running":
                if node_name == local_name:
                    return "127.0.0.1", proc.port
                node_cfg = next(
                    (n for n in config.nodes if n.name == node_name), None
                )
                host = node_cfg.host if node_cfg else "127.0.0.1"
                return host, proc.port

    raise ValueError(
        f"Model '{model_name}' not found in config or running processes"
    )


def run_flow(flow: FlowDefinition, input_data: Any = None) -> FlowResult:
    """Execute a flow definition.

    This is a basic sequential executor. When Prefect is installed,
    flows can be compiled to Prefect flows for advanced features
    (retries, caching, parallel execution, scheduling).

    Args:
        flow: The flow definition to execute.
        input_data: Initial input data.

    Returns:
        FlowResult with all step results.
    """
    results: dict[str, Any] = {}
    step_results: list[StepResult] = []

    for step in flow.steps:
        try:
            # Resolve input references
            step_input = _resolve_input(step.input, input_data, results)

            # Execute step
            output = _execute_step(step, step_input)

            results[step.id] = output
            step_results.append(StepResult(
                step_id=step.id,
                success=True,
                output=output,
            ))

        except Exception as e:
            step_results.append(StepResult(
                step_id=step.id,
                success=False,
                error=str(e),
            ))
            return FlowResult(
                flow_name=flow.name,
                success=False,
                step_results=step_results,
                error=f"Step '{step.id}' failed: {e}",
            )

    # Resolve final output
    final_output = _resolve_input(flow.output, input_data, results) if flow.output else None

    return FlowResult(
        flow_name=flow.name,
        success=True,
        step_results=step_results,
        final_output=final_output,
    )


def _resolve_input(
    input_ref: str | None,
    input_data: Any,
    results: dict[str, Any],
) -> Any:
    """Resolve input references like $input or $steps.transcribe.output."""
    if input_ref is None:
        return input_data

    if input_ref == "$input":
        return input_data

    if input_ref.startswith("$steps."):
        parts = input_ref.split(".")
        if len(parts) >= 3:
            step_id = parts[1]
            return results.get(step_id)

    # Resolve embedded references in template strings
    # e.g., "Summarize: {{ steps.transcribe.output }}"
    if "{{" in input_ref and "}}" in input_ref:
        import re

        def _replace(match: re.Match) -> str:
            expr = match.group(1).strip()
            if expr.startswith("steps."):
                parts = expr.split(".")
                if len(parts) >= 2:
                    step_id = parts[1]
                    val = results.get(step_id)
                    return str(val) if val is not None else ""
            return match.group(0)

        return re.sub(r"\{\{\s*(.*?)\s*\}\}", _replace, input_ref)

    return input_ref


def _execute_step(step: FlowStep, input_data: Any) -> Any:
    """Execute a single flow step by dispatching to the appropriate handler."""
    if step.type == "audio.transcribe":
        return _execute_stt(step, input_data)
    if step.type == "text.generate":
        return _execute_chat(step, input_data)
    if step.type == "fs.glob":
        return _execute_glob(input_data)
    if step.type == "fs.write":
        return _execute_write(input_data)

    raise ValueError(f"Unknown step type: {step.type}")


def _execute_stt(step: FlowStep, input_data: Any) -> str:
    """Execute an STT step by calling the model's OpenAI-compatible API."""
    if not step.model:
        raise ValueError("STT step requires a 'model' field")

    host, port = _find_model_endpoint(step.model, node_hint=step.node)
    url = f"http://{host}:{port}/v1/audio/transcriptions"

    # input_data should be audio bytes or a file path
    if isinstance(input_data, (str, Path)):
        audio_path = Path(input_data)
        if audio_path.exists():
            audio_data = audio_path.read_bytes()
            filename = audio_path.name
        else:
            raise ValueError(f"Audio file not found: {input_data}")
    elif isinstance(input_data, bytes):
        audio_data = input_data
        filename = "audio.wav"
    else:
        raise ValueError(
            f"STT step expects audio file path or bytes, got {type(input_data).__name__}"
        )

    # Build multipart form data
    boundary = "----SiloFlowBoundary"
    body = io.BytesIO()

    # model field
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    body.write(f"{step.model}\r\n".encode())

    # file field
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    body.write(b"Content-Type: application/octet-stream\r\n\r\n")
    body.write(audio_data)
    body.write(b"\r\n")

    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result.get("text", "")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        raise RuntimeError(f"STT request failed ({e.code}): {error_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach model server at {url}: {e.reason}"
        ) from e


def _execute_chat(step: FlowStep, input_data: Any) -> str:
    """Execute a chat step by calling the model's OpenAI-compatible API."""
    if not step.model:
        raise ValueError("Chat step requires a 'model' field")

    host, port = _find_model_endpoint(step.model, node_hint=step.node)
    url = f"http://{host}:{port}/v1/chat/completions"

    # Build the prompt from input
    prompt = str(input_data) if input_data is not None else ""

    payload = json.dumps({
        "model": step.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        raise RuntimeError(f"Chat request failed ({e.code}): {error_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach model server at {url}: {e.reason}"
        ) from e


def _execute_glob(input_data: Any) -> list[str]:
    """Execute a filesystem glob step."""
    pattern = str(input_data) if input_data else "*"
    return [str(p) for p in Path(".").glob(pattern)]


def _execute_write(input_data: Any) -> str:
    """Execute a filesystem write step."""
    if isinstance(input_data, dict):
        content = input_data.get("content", "")
        path = input_data.get("path", "output.txt")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(str(content))
        return path
    raise ValueError("Write step requires dict input with 'content' and 'path'")
