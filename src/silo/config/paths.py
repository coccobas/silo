"""Canonical paths for Silo configuration and state."""

from pathlib import Path

CONFIG_DIR = Path.home() / ".silo"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
REGISTRY_PATH = CONFIG_DIR / "registry.json"
LOGS_DIR = CONFIG_DIR / "logs"
PIDS_DIR = CONFIG_DIR / "pids"


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for directory in (CONFIG_DIR, LOGS_DIR, PIDS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
