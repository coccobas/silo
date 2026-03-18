"""Process manager — spawn and manage model server subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from silo.config.paths import LOGS_DIR, ensure_dirs
from silo.process.pid import (
    is_running,
    list_pid_entries,
    list_pids,
    read_pid,
    read_pid_entry,
    remove_pid,
    write_pid,
)


@dataclass(frozen=True)
class SpawnResult:
    """Result of spawning a model process."""

    pid: int
    instance_id: str


@dataclass(frozen=True)
class ProcessInfo:
    """Information about a running model process."""

    name: str
    pid: int
    port: int
    repo_id: str
    status: str  # "running", "stopped", "unknown"
    instance_id: str = ""


def spawn_model(
    name: str,
    repo_id: str,
    host: str = "127.0.0.1",
    port: int = 8800,
    quantize: str | None = None,
    output: str | None = None,
    pids_dir: Path | None = None,
    logs_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> SpawnResult:
    """Spawn a `silo serve` subprocess for a model.

    Args:
        name: Friendly model name.
        repo_id: HuggingFace repository ID.
        host: Bind host.
        port: Bind port.
        quantize: Optional quantization level.
        output: Optional output path for conversion.
        pids_dir: Override PID directory.
        logs_dir: Override logs directory.
        extra_env: Extra environment variables for the subprocess.

    Returns:
        SpawnResult with PID and instance_id.
    """
    import uuid

    ensure_dirs()
    target_logs = logs_dir or LOGS_DIR
    target_logs.mkdir(parents=True, exist_ok=True)

    log_file = target_logs / f"{name}.log"
    instance_id = str(uuid.uuid4())

    # Use the installed entry point (same venv as current process)
    entry_point = Path(sys.executable).parent / "silo"
    cmd = [
        str(entry_point),
        "serve", repo_id,
        "--host", host,
        "--port", str(port),
        "--name", name,
    ]
    if quantize:
        cmd.extend(["--quantize", quantize])
    if output:
        cmd.extend(["--output", output])

    env = {**os.environ, **(extra_env or {})}

    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    write_pid(
        name,
        proc.pid,
        port=port,
        host=host,
        repo_id=repo_id,
        instance_id=instance_id,
        pids_dir=pids_dir,
    )
    return SpawnResult(pid=proc.pid, instance_id=instance_id)


def stop_model(
    name: str,
    grace_period: int = 30,
    pids_dir: Path | None = None,
) -> bool:
    """Stop a model process gracefully.

    Sends SIGTERM, waits up to grace_period seconds,
    then SIGKILL if still running.

    Args:
        name: Model name.
        grace_period: Seconds to wait before SIGKILL.
        pids_dir: Override PID directory.

    Returns:
        True if the process was stopped, False if not found.
    """
    pid = read_pid(name, pids_dir=pids_dir)
    if pid is None:
        return False

    if not is_running(pid):
        remove_pid(name, pids_dir=pids_dir)
        return True

    # Send SIGTERM
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        remove_pid(name, pids_dir=pids_dir)
        return True

    # Wait for graceful shutdown
    deadline = time.monotonic() + grace_period
    while time.monotonic() < deadline:
        if not is_running(pid):
            remove_pid(name, pids_dir=pids_dir)
            return True
        time.sleep(0.1)

    # Force kill
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    remove_pid(name, pids_dir=pids_dir)
    return True


def get_status(
    name: str,
    port: int = 0,
    repo_id: str = "",
    pids_dir: Path | None = None,
) -> ProcessInfo:
    """Get the status of a model process.

    Args:
        name: Model name.
        port: Expected port (for display).
        repo_id: Repository ID (for display).
        pids_dir: Override PID directory.

    Returns:
        ProcessInfo with current status.
    """
    entry = read_pid_entry(name, pids_dir=pids_dir)
    if entry is None:
        return ProcessInfo(name=name, pid=0, port=port, repo_id=repo_id, status="stopped")

    if is_running(entry.pid):
        return ProcessInfo(
            name=name, pid=entry.pid, port=entry.port or port,
            repo_id=entry.repo_id or repo_id, status="running",
            instance_id=entry.instance_id,
        )

    # Stale PID file
    remove_pid(name, pids_dir=pids_dir)
    return ProcessInfo(name=name, pid=0, port=port, repo_id=repo_id, status="stopped")


def list_running(pids_dir: Path | None = None) -> list[ProcessInfo]:
    """List all model processes with their status.

    Returns:
        List of ProcessInfo for all known processes.
    """
    entries = list_pid_entries(pids_dir=pids_dir)
    result: list[ProcessInfo] = []
    for name, entry in entries.items():
        status = "running" if is_running(entry.pid) else "stopped"
        if status == "stopped":
            remove_pid(name, pids_dir=pids_dir)
        result.append(
            ProcessInfo(
                name=name,
                pid=entry.pid,
                port=entry.port,
                repo_id=entry.repo_id,
                status=status,
                instance_id=entry.instance_id,
            )
        )
    return result
