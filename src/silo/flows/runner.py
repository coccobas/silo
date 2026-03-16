"""Flow execution engine — runs flow definitions."""

from __future__ import annotations

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

    return input_ref


def _execute_step(step: FlowStep, input_data: Any) -> Any:
    """Execute a single flow step.

    This is a stub executor that dispatches to the appropriate
    backend based on step type. In a real implementation, this
    would call the actual model APIs.
    """
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
    """Execute an STT step by calling the local model API."""

    if not step.model:
        raise ValueError("STT step requires a 'model' field")

    # In a real implementation, this would determine the port from config
    # For now, this is a placeholder
    raise NotImplementedError(
        f"STT execution for model '{step.model}' not yet connected. "
        "Use 'silo serve' to start the model first."
    )


def _execute_chat(step: FlowStep, input_data: Any) -> str:
    """Execute a chat step by calling the local model API."""
    if not step.model:
        raise ValueError("Chat step requires a 'model' field")

    raise NotImplementedError(
        f"Chat execution for model '{step.model}' not yet connected. "
        "Use 'silo serve' to start the model first."
    )


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
