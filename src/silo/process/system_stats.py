"""CPU and GPU usage monitoring for Apple Silicon."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class SystemStats:
    """CPU and GPU usage information."""

    cpu_percent: float  # Total CPU usage (user + sys)
    gpu_percent: float  # GPU utilization (0.0 if unavailable)
    gpu_name: str  # e.g. "Apple M2 Max" or "unknown"


def get_system_stats() -> SystemStats:
    """Get current CPU and GPU usage.

    Uses macOS-specific tools. Returns zeros on non-Darwin systems.
    """
    if platform.system() != "Darwin":
        return SystemStats(cpu_percent=0.0, gpu_percent=0.0, gpu_name="unknown")

    cpu = _get_cpu_usage()
    gpu_pct = _get_gpu_usage()
    gpu_name = _get_gpu_name()

    return SystemStats(
        cpu_percent=round(cpu, 1),
        gpu_percent=round(gpu_pct, 1),
        gpu_name=gpu_name,
    )


def _get_cpu_usage() -> float:
    """Parse CPU usage from top.

    Runs ``top -l 1 -n 0 -s 0`` and extracts user + sys percentages.
    """
    try:
        result = subprocess.run(
            ["top", "-l", "1", "-n", "0", "-s", "0"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if line.startswith("CPU usage:"):
                # "CPU usage: 5.55% user, 3.33% sys, 91.11% idle"
                parts = line.split(",")
                user = 0.0
                sys_ = 0.0
                for part in parts:
                    part = part.strip()
                    if "user" in part:
                        user = float(part.split("%")[0].split()[-1])
                    elif "sys" in part:
                        sys_ = float(part.split("%")[0].split()[-1])
                return user + sys_
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return 0.0


def _get_gpu_usage() -> float:
    """Read GPU utilization from IOKit via ioreg.

    Parses the ``PerformanceStatistics`` dict from ``AGXAccelerator``
    looking for ``Device Utilization %``, ``Renderer Utilization %``,
    or ``GPU Activity(%)``.
    Returns 0.0 if the metric is unavailable.
    """
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "AGXAccelerator"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        import re

        output = result.stdout
        # Look for "Device Utilization %" = <number> inside PerformanceStatistics
        for key in (
            "Device Utilization %",
            "Renderer Utilization %",
            "GPU Activity(%)",
        ):
            pattern = re.escape(f'"{key}"') + r"\s*=\s*(\d+)"
            match = re.search(pattern, output)
            if match:
                return float(match.group(1))
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return 0.0


def _get_gpu_name() -> str:
    """Get the GPU/chip name via sysctl."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        brand = result.stdout.strip()
        # "Apple M2 Max" → keep as-is
        if brand:
            return brand
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return "unknown"
