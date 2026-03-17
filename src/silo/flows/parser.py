"""YAML flow parser — convert flow definitions to executable steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FlowStep:
    """A single step in a flow pipeline."""

    id: str
    type: str  # e.g., "audio.transcribe", "text.generate", "fs.glob", "fs.write"
    model: str | None = None
    input: str | None = None
    map: bool = False


@dataclass(frozen=True)
class FlowConfig:
    """Configuration for retry, concurrency, caching."""

    retries: int = 0
    retry_delay: int = 30
    concurrency: int = 1
    cache: bool = False


@dataclass(frozen=True)
class FlowDefinition:
    """A parsed flow definition."""

    name: str
    description: str = ""
    schedule: str | None = None
    config: FlowConfig = field(default_factory=FlowConfig)
    steps: list[FlowStep] = field(default_factory=list)
    output: str | None = None


def parse_flow(path: Path) -> FlowDefinition:
    """Parse a YAML flow definition file.

    Args:
        path: Path to the YAML flow file.

    Returns:
        Parsed FlowDefinition.

    Raises:
        ValueError: If the flow definition is invalid.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid flow file: expected a YAML mapping, got {type(data).__name__}")

    name = data.get("name")
    if not name:
        raise ValueError("Flow must have a 'name' field")

    steps_data = data.get("steps", [])
    if not steps_data:
        raise ValueError("Flow must have at least one step")

    steps: list[FlowStep] = []
    for step_data in steps_data:
        if not isinstance(step_data, dict):
            raise ValueError(f"Each step must be a mapping, got {type(step_data).__name__}")

        step_id = step_data.get("id")
        if not step_id:
            raise ValueError("Each step must have an 'id' field")

        step_type = step_data.get("type", "")
        step_input = step_data.get("input")
        if isinstance(step_input, dict):
            step_input = yaml.dump(step_input, default_flow_style=True).strip()

        steps.append(FlowStep(
            id=step_id,
            type=step_type,
            model=step_data.get("model"),
            input=step_input,
            map=step_data.get("map", False),
        ))

    config_data = data.get("config", {})
    config = FlowConfig(
        retries=config_data.get("retries", 0),
        retry_delay=config_data.get("retry_delay", 30),
        concurrency=config_data.get("concurrency", 1),
        cache=config_data.get("cache", False),
    )

    return FlowDefinition(
        name=name,
        description=data.get("description", ""),
        schedule=data.get("schedule"),
        config=config,
        steps=steps,
        output=data.get("output"),
    )


def save_flow(flow: FlowDefinition, flows_dir: Path) -> Path:
    """Save a flow definition to a YAML file.

    Args:
        flow: The flow definition to save.
        flows_dir: Directory to save the YAML file in.

    Returns:
        Path to the saved file.
    """
    flows_dir.mkdir(parents=True, exist_ok=True)
    path = flows_dir / f"{flow.name}.yaml"

    data: dict = {"name": flow.name}
    if flow.description:
        data["description"] = flow.description

    steps_data = []
    for step in flow.steps:
        step_dict: dict = {"id": step.id, "type": step.type}
        if step.model:
            step_dict["model"] = step.model
        if step.input:
            step_dict["input"] = step.input
        if step.map:
            step_dict["map"] = True
        steps_data.append(step_dict)
    data["steps"] = steps_data

    if flow.output:
        data["output"] = flow.output

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return path


def list_flows(flows_dir: Path | None = None) -> list[FlowDefinition]:
    """List all flow definitions in a directory.

    Args:
        flows_dir: Directory to scan for YAML flow files.

    Returns:
        List of parsed FlowDefinitions.
    """
    if flows_dir is None or not flows_dir.exists():
        return []

    result: list[FlowDefinition] = []
    for path in sorted(flows_dir.glob("*.yaml")):
        try:
            result.append(parse_flow(path))
        except (ValueError, yaml.YAMLError):
            continue
    return result
