"""PID file management for model server processes.

Each process is tracked via a JSON manifest (``<name>.pid``) that stores the
PID along with metadata (port, repo_id, host, runtime).  Legacy plain-text
PID files (just a number) are read transparently for backwards compatibility.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from silo.config.paths import PIDS_DIR, ensure_dirs


@dataclass(frozen=True)
class PidEntry:
    """Metadata stored alongside a process PID."""

    pid: int
    port: int = 0
    host: str = "127.0.0.1"
    repo_id: str = ""
    runtime: str = ""
    instance_id: str = ""


def write_pid(
    name: str,
    pid: int,
    *,
    port: int = 0,
    host: str = "127.0.0.1",
    repo_id: str = "",
    runtime: str = "",
    instance_id: str = "",
    pids_dir: Path | None = None,
) -> Path:
    """Write a PID manifest for a named model process.

    Auto-generates an instance_id (UUID4) if not provided.

    Returns:
        Path to the PID file.
    """
    import uuid

    target_dir = pids_dir or PIDS_DIR
    ensure_dirs()
    target_dir.mkdir(parents=True, exist_ok=True)
    pid_file = target_dir / f"{name}.pid"
    iid = instance_id or str(uuid.uuid4())
    entry = PidEntry(
        pid=pid, port=port, host=host, repo_id=repo_id,
        runtime=runtime, instance_id=iid,
    )
    pid_file.write_text(json.dumps(asdict(entry)))
    return pid_file


def read_pid(name: str, pids_dir: Path | None = None) -> int | None:
    """Read a PID from a named PID file.

    Returns:
        The PID as int, or None if no PID file exists.
    """
    entry = read_pid_entry(name, pids_dir=pids_dir)
    return entry.pid if entry else None


def read_pid_entry(name: str, pids_dir: Path | None = None) -> PidEntry | None:
    """Read a full PID entry (with metadata) from a named PID file.

    Handles both JSON manifests and legacy plain-text PID files.
    """
    target_dir = pids_dir or PIDS_DIR
    pid_file = target_dir / f"{name}.pid"
    if not pid_file.exists():
        return None
    try:
        raw = pid_file.read_text().strip()
        # Try JSON manifest first
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return PidEntry(**{k: v for k, v in data.items() if k in PidEntry.__dataclass_fields__})
            # JSON-parsed plain integer (e.g. "42" → 42)
            if isinstance(data, int):
                return PidEntry(pid=data)
        except (json.JSONDecodeError, TypeError):
            pass
        # Legacy: plain integer string
        return PidEntry(pid=int(raw))
    except (ValueError, OSError):
        return None


def remove_pid(name: str, pids_dir: Path | None = None) -> None:
    """Remove a PID file."""
    target_dir = pids_dir or PIDS_DIR
    pid_file = target_dir / f"{name}.pid"
    pid_file.unlink(missing_ok=True)


def is_running(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def list_pids(pids_dir: Path | None = None) -> dict[str, int]:
    """List all named model processes from PID files.

    Returns:
        Dict mapping model name to PID.
    """
    target_dir = pids_dir or PIDS_DIR
    if not target_dir.exists():
        return {}

    result: dict[str, int] = {}
    for pid_file in target_dir.glob("*.pid"):
        name = pid_file.stem
        pid = read_pid(name, pids_dir=target_dir)
        if pid is not None:
            result[name] = pid
    return result


def list_pid_entries(pids_dir: Path | None = None) -> dict[str, PidEntry]:
    """List all named model processes with full metadata.

    Returns:
        Dict mapping model name to PidEntry.
    """
    target_dir = pids_dir or PIDS_DIR
    if not target_dir.exists():
        return {}

    result: dict[str, PidEntry] = {}
    for pid_file in target_dir.glob("*.pid"):
        name = pid_file.stem
        entry = read_pid_entry(name, pids_dir=target_dir)
        if entry is not None:
            result[name] = entry
    return result
