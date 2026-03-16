"""Pydantic configuration models for Silo."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class RestartPolicy(StrEnum):
    ALWAYS = "always"
    ON_FAILURE = "on-failure"
    NEVER = "never"


class NodeConfig(BaseModel):
    """Configuration for a remote Silo agent node."""

    model_config = {"frozen": True}

    name: str
    host: str
    port: int = 9900


class ModelConfig(BaseModel):
    """Configuration for a single model instance."""

    model_config = {"frozen": True}

    name: str
    repo: str
    host: str = "127.0.0.1"
    port: int = 8800
    backend: str | None = None
    quantize: str | None = None
    output: str | None = None
    warmup: bool = False
    restart: RestartPolicy = RestartPolicy.ON_FAILURE
    timeout: int = 120
    node: str | None = None


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = {"frozen": True}

    nodes: list[NodeConfig] = Field(default_factory=list)
    models: list[ModelConfig] = Field(default_factory=list)
