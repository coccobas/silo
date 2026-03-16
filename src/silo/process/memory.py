"""Memory monitoring for Apple Silicon unified memory."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryInfo:
    """System memory information."""

    total_gb: float
    available_gb: float
    used_gb: float
    pressure: str  # "normal", "warn", "critical"

    @property
    def usage_percent(self) -> float:
        """Memory usage as a percentage."""
        if self.total_gb == 0:
            return 0.0
        return (self.used_gb / self.total_gb) * 100


def get_memory_info() -> MemoryInfo:
    """Get current system memory information.

    Uses macOS sysctl for Apple Silicon unified memory.
    Falls back to psutil-style parsing for other systems.

    Returns:
        MemoryInfo with current memory stats.
    """
    if platform.system() != "Darwin":
        return MemoryInfo(total_gb=0, available_gb=0, used_gb=0, pressure="unknown")

    try:
        # Get total memory
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        total_bytes = int(result.stdout.strip())
        total_gb = total_bytes / (1024**3)

        # Get memory pressure
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout.lower()

        # Parse free pages from vm_stat as a rough estimate
        vm_result = subprocess.run(
            ["vm_stat"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        free_pages = 0
        inactive_pages = 0
        page_size = 16384  # Default for Apple Silicon

        for line in vm_result.stdout.splitlines():
            if "page size" in line.lower():
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        page_size = int(part)
            elif line.startswith("Pages free:"):
                free_pages = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages inactive:"):
                inactive_pages = int(line.split(":")[1].strip().rstrip("."))

        available_bytes = (free_pages + inactive_pages) * page_size
        available_gb = available_bytes / (1024**3)
        used_gb = total_gb - available_gb

        # Determine pressure level
        if "critical" in output:
            pressure = "critical"
        elif "warn" in output:
            pressure = "warn"
        else:
            pressure = "normal"

        return MemoryInfo(
            total_gb=round(total_gb, 1),
            available_gb=round(available_gb, 1),
            used_gb=round(used_gb, 1),
            pressure=pressure,
        )

    except (subprocess.SubprocessError, ValueError, OSError):
        return MemoryInfo(total_gb=0, available_gb=0, used_gb=0, pressure="unknown")


def check_memory_pressure(threshold_percent: float = 85.0) -> bool:
    """Check if memory usage is above the threshold.

    Args:
        threshold_percent: Warning threshold (default 85%).

    Returns:
        True if memory usage is above threshold.
    """
    info = get_memory_info()
    return info.usage_percent > threshold_percent
