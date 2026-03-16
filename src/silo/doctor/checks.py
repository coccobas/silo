"""Diagnostic checks for Silo environment."""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from enum import StrEnum


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    message: str


def check_python() -> CheckResult:
    version = sys.version_info
    ver_str = f"{version.major}.{version.minor}.{version.micro}"
    if version >= (3, 12):
        return CheckResult("Python", CheckStatus.OK, ver_str)
    return CheckResult("Python", CheckStatus.FAIL, f"{ver_str} (need 3.12+)")


def check_apple_silicon() -> CheckResult:
    machine = platform.machine()
    if machine == "arm64" and platform.system() == "Darwin":
        return CheckResult("Apple Silicon", CheckStatus.OK, f"{machine}")
    if platform.system() != "Darwin":
        return CheckResult("Apple Silicon", CheckStatus.FAIL, f"Not macOS ({platform.system()})")
    return CheckResult("Apple Silicon", CheckStatus.WARN, f"Architecture: {machine}")


def check_mlx() -> CheckResult:
    try:
        import mlx  # type: ignore[import-untyped]

        version = getattr(mlx, "__version__", "installed")
        return CheckResult("MLX", CheckStatus.OK, f"v{version}")
    except ImportError:
        return CheckResult("MLX", CheckStatus.WARN, "Not installed (uv pip install mlx)")


def check_mlx_lm() -> CheckResult:
    try:
        import mlx_lm  # type: ignore[import-untyped]

        version = getattr(mlx_lm, "__version__", "installed")
        return CheckResult("mlx-lm", CheckStatus.OK, f"v{version}")
    except ImportError:
        return CheckResult(
            "mlx-lm", CheckStatus.WARN, "Not installed (uv pip install 'silo[mlx]')"
        )


def check_huggingface_hub() -> CheckResult:
    try:
        import huggingface_hub

        return CheckResult("huggingface-hub", CheckStatus.OK, f"v{huggingface_hub.__version__}")
    except ImportError:
        return CheckResult("huggingface-hub", CheckStatus.FAIL, "Not installed")


def check_ffmpeg() -> CheckResult:
    if shutil.which("ffmpeg"):
        return CheckResult("ffmpeg", CheckStatus.OK, "Available")
    return CheckResult("ffmpeg", CheckStatus.WARN, "Not found (needed for non-WAV audio)")


def check_memory() -> CheckResult:
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        mem_bytes = int(result.stdout.strip())
        mem_gb = mem_bytes / (1024**3)
        status = CheckStatus.OK if mem_gb >= 8 else CheckStatus.WARN
        return CheckResult("Memory", status, f"{mem_gb:.0f} GB unified memory")
    except Exception:
        return CheckResult("Memory", CheckStatus.WARN, "Could not detect")


def check_registry() -> CheckResult:
    from silo.config.paths import REGISTRY_PATH

    if not REGISTRY_PATH.exists():
        return CheckResult(
            "Registry", CheckStatus.OK, "Not yet created (will be created on first use)"
        )

    try:
        from silo.registry.store import Registry

        registry = Registry.load()
        count = len(registry.list())
        return CheckResult("Registry", CheckStatus.OK, f"{count} model(s) registered")
    except Exception as e:
        return CheckResult("Registry", CheckStatus.FAIL, f"Corrupt: {e}")


def run_all_checks() -> list[CheckResult]:
    return [
        check_python(),
        check_apple_silicon(),
        check_mlx(),
        check_mlx_lm(),
        check_huggingface_hub(),
        check_ffmpeg(),
        check_memory(),
        check_registry(),
    ]
