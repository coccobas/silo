"""Canonical paths for Silo configuration and state."""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".silo"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
REGISTRY_PATH = CONFIG_DIR / "registry.json"
LOGS_DIR = CONFIG_DIR / "logs"
PIDS_DIR = CONFIG_DIR / "pids"
WAKE_MODELS_DIR = CONFIG_DIR / "wake-models"
CLUSTER_WORKERS_PATH = CONFIG_DIR / "cluster_workers.json"
AGENT_LOCK_PATH = CONFIG_DIR / "agent.lock"


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for directory in (CONFIG_DIR, LOGS_DIR, PIDS_DIR, WAKE_MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def acquire_agent_lock(pid: int, port: int) -> bool:
    """Try to acquire the agent singleton lock.

    Returns True if the lock was acquired.  Returns False if another
    agent is already running (the lock file exists and the PID is alive).
    """
    ensure_dirs()
    if AGENT_LOCK_PATH.exists():
        try:
            data = json.loads(AGENT_LOCK_PATH.read_text())
            existing_pid = data.get("pid", 0)
            # Check if the process is still alive
            os.kill(existing_pid, 0)
            # Process exists — lock is held
            return False
        except (ProcessLookupError, OSError, json.JSONDecodeError, ValueError):
            # Process is dead or lock is corrupt — safe to take over
            pass

    AGENT_LOCK_PATH.write_text(json.dumps({"pid": pid, "port": port}))
    return True


def read_agent_lock() -> dict | None:
    """Read the agent lock file.  Returns {"pid": ..., "port": ...} or None."""
    if not AGENT_LOCK_PATH.exists():
        return None
    try:
        data = json.loads(AGENT_LOCK_PATH.read_text())
        existing_pid = data.get("pid", 0)
        os.kill(existing_pid, 0)
        return data
    except (ProcessLookupError, OSError, json.JSONDecodeError, ValueError):
        return None


def release_agent_lock() -> None:
    """Release the agent singleton lock."""
    AGENT_LOCK_PATH.unlink(missing_ok=True)
