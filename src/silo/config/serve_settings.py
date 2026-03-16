"""Persistent per-model serve settings stored at ~/.silo/serve_settings.json."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from silo.config.paths import CONFIG_DIR

if TYPE_CHECKING:
    from silo.tui.widgets.serve_modal import ServeSettings

SERVE_SETTINGS_PATH = CONFIG_DIR / "serve_settings.json"


def _load_all() -> dict[str, dict]:
    """Load the entire settings file."""
    if not SERVE_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SERVE_SETTINGS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_all(data: dict[str, dict]) -> None:
    """Atomic write of the settings file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SERVE_SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(SERVE_SETTINGS_PATH)


def save_settings(repo_id: str, settings: ServeSettings) -> None:
    """Persist serve settings for a model."""
    data = _load_all()
    data[repo_id] = asdict(settings)
    _save_all(data)


def load_settings(repo_id: str) -> dict | None:
    """Load saved serve settings for a model, or None if not found."""
    data = _load_all()
    return data.get(repo_id)


def remove_settings(repo_id: str) -> None:
    """Remove saved settings for a model."""
    data = _load_all()
    if repo_id in data:
        del data[repo_id]
        _save_all(data)
