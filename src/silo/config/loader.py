"""Configuration loader with YAML parsing and env var overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from silo.config.models import AppConfig, LitellmConfig, ModelConfig
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

    model_overrides: dict[str, str | int] = {}
    if env_host:
        model_overrides["host"] = env_host
    if env_port:
        model_overrides["port"] = int(env_port)

    updated_models = (
        [ModelConfig(**{**m.model_dump(), **model_overrides}) for m in config.models]
        if model_overrides
        else list(config.models)
    )

    # LiteLLM env overrides
    litellm_url = os.environ.get("SILO_LITELLM_URL")
    litellm_key = os.environ.get("SILO_LITELLM_API_KEY")
    litellm_enabled = os.environ.get("SILO_LITELLM_ENABLED")

    if litellm_url or litellm_key or litellm_enabled is not None:
        litellm_overrides: dict[str, object] = {**config.litellm.model_dump()}
        if litellm_url:
            litellm_overrides["url"] = litellm_url
        if litellm_key:
            litellm_overrides["api_key"] = litellm_key
        if litellm_enabled is not None:
            litellm_overrides["enabled"] = litellm_enabled.lower() in ("1", "true", "yes")
        updated_litellm = LitellmConfig(**litellm_overrides)
    else:
        updated_litellm = config.litellm

    if not model_overrides and updated_litellm is config.litellm:
        return config

    return AppConfig(
        nodes=list(config.nodes),
        models=updated_models,
        litellm=updated_litellm,
    )
