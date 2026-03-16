"""Configuration loader with YAML parsing and env var overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from silo.config.models import AppConfig, ModelConfig
from silo.config.paths import CONFIG_FILE


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from YAML file with env var overrides.

    Args:
        path: Path to config file. Defaults to ~/.silo/config.yaml.

    Returns:
        Validated AppConfig instance.
    """
    config_path = path or CONFIG_FILE

    if not config_path.exists():
        return _apply_env_overrides(AppConfig())

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return _apply_env_overrides(AppConfig())

    config = AppConfig.model_validate(raw)
    return _apply_env_overrides(config)


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    """Apply SILO_* environment variable overrides to config."""
    env_host = os.environ.get("SILO_HOST")
    env_port = os.environ.get("SILO_PORT")

    if not env_host and not env_port:
        return config

    overrides: dict[str, str | int] = {}
    if env_host:
        overrides["host"] = env_host
    if env_port:
        overrides["port"] = int(env_port)

    updated_models = [
        ModelConfig(**{**m.model_dump(), **overrides}) for m in config.models
    ]
    return AppConfig(models=updated_models)
