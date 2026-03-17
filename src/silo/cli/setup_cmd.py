"""CLI: setup command — install optional dependency groups."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from silo.cli.app import app

console = Console()

setup_app = typer.Typer(name="setup", help="Install optional dependency groups.")
app.add_typer(setup_app)

# Map feature names to their packages and any special install instructions.
# Packages listed here match pyproject.toml [project.optional-dependencies].
_FEATURES: dict[str, dict] = {
    "wake": {
        "packages": ["sounddevice>=0.5", "numpy>=1.26", "onnxruntime>=1.16"],
        "post_install": [
            # openwakeword depends on tflite-runtime which has no cp312+ wheels.
            # We use onnxruntime for inference, so install without its deps.
            {"packages": ["openwakeword"], "no_deps": True},
        ],
        "verify": ["openwakeword", "sounddevice", "numpy", "onnxruntime"],
        "description": "Wake word detection (microphone + openwakeword)",
    },
    "tui": {
        "packages": ["textual>=0.50"],
        "verify": ["textual"],
        "description": "Terminal UI dashboard",
    },
    "mlx": {
        "packages": ["mlx>=0.22", "mlx-lm>=0.22"],
        "verify": ["mlx", "mlx_lm"],
        "description": "MLX backend for Apple Silicon",
    },
    "audio": {
        "packages": ["mlx>=0.22", "mlx-whisper>=0.4", "mlx-audio>=0.2"],
        "verify": ["mlx_whisper", "mlx_audio"],
        "description": "Speech-to-text and text-to-speech (MLX Audio)",
    },
    "llamacpp": {
        "packages": ["llama-cpp-python>=0.3"],
        "verify": ["llama_cpp"],
        "description": "llama.cpp backend (GGUF models)",
    },
    "all": {
        "packages": [],
        "install_features": ["wake", "tui", "mlx", "audio", "llamacpp"],
        "verify": [],
        "description": "Everything",
    },
}


def _find_installer() -> list[str]:
    """Find the best pip installer available."""
    # Prefer uv pip if available (faster)
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install"]
    # Fall back to python -m pip
    return [sys.executable, "-m", "pip", "install"]


def _install_packages(
    packages: list[str],
    no_deps: bool = False,
    upgrade: bool = True,
) -> tuple[bool, str]:
    """Install packages and return (success, error_output)."""
    if not packages:
        return True, ""

    cmd = [*_find_installer(), "--quiet"]
    if upgrade:
        cmd.append("--upgrade")
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(packages)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, ""


def _verify_imports(modules: list[str]) -> list[str]:
    """Return list of modules that failed to import."""
    failed = []
    for mod in modules:
        try:
            __import__(mod)
        except ImportError:
            failed.append(mod)
    return failed


def _install_feature(name: str) -> bool:
    """Install a single feature. Returns True on success."""
    info = _FEATURES[name]

    # Install main packages
    packages = info.get("packages", [])
    if packages:
        console.print(f"  [cyan]Installing {', '.join(packages)}...[/cyan]")
        ok, err = _install_packages(packages)
        if not ok:
            console.print(f"  [red]Failed: {err}[/red]")
            return False

    # Post-install packages (e.g., openwakeword --no-deps)
    for step in info.get("post_install", []):
        pkgs = step["packages"]
        no_deps = step.get("no_deps", False)
        label = " (--no-deps)" if no_deps else ""
        console.print(f"  [cyan]Installing {', '.join(pkgs)}{label}...[/cyan]")
        ok, err = _install_packages(pkgs, no_deps=no_deps)
        if not ok:
            console.print(f"  [red]Failed: {err}[/red]")
            return False

    return True


@setup_app.command("list")
def setup_list() -> None:
    """Show available features and their install status."""
    from rich.table import Table

    table = Table(title="Available Features")
    table.add_column("FEATURE", style="bold")
    table.add_column("DESCRIPTION")
    table.add_column("STATUS")

    for name, info in _FEATURES.items():
        if name == "all":
            continue
        verify = info.get("verify", [])
        if verify:
            failed = _verify_imports(verify)
            if not failed:
                status = "[green]installed[/]"
            else:
                status = "[dim]not installed[/]"
        else:
            status = "[dim]—[/]"
        table.add_row(name, info["description"], status)

    console.print(table)
    console.print("\n[dim]Install with: silo setup install <feature>[/]")


@setup_app.command("install")
def setup_install(
    feature: str = typer.Argument(
        help="Feature to install (wake, tui, mlx, audio, llamacpp, all)",
    ),
) -> None:
    """Install a feature and its dependencies."""
    if feature not in _FEATURES:
        console.print(f"[red]Unknown feature: '{feature}'[/red]")
        console.print(f"[dim]Available: {', '.join(_FEATURES)}[/dim]")
        raise typer.Exit(1)

    info = _FEATURES[feature]

    # "all" delegates to individual features
    features_to_install = info.get("install_features", [feature])

    for feat_name in features_to_install:
        console.print(f"[bold]Setting up {feat_name}...[/bold]")
        if not _install_feature(feat_name):
            console.print(f"[red]Failed to install '{feat_name}'.[/red]")
            raise typer.Exit(1)

    # Verify
    all_verify: list[str] = []
    for feat_name in features_to_install:
        all_verify.extend(_FEATURES[feat_name].get("verify", []))

    if all_verify:
        failed = _verify_imports(all_verify)
        if failed:
            console.print(
                f"[yellow]Warning: could not import: {', '.join(failed)}[/yellow]"
            )
        else:
            console.print(f"[green]'{feature}' installed and verified.[/green]")
    else:
        console.print(f"[green]'{feature}' installed.[/green]")
