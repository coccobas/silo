"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from silo.cli.app import app


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def cli_app():
    # Ensure sub-commands are registered
    from silo.cli import (  # noqa: F401
        agent_cmd,
        convert_cmd,
        doctor_cmd,
        down_cmd,
        flow_cmd,
        init_cmd,
        logs_cmd,
        models_cmd,
        ps_cmd,
        run_cmd,
        serve_cmd,
        ui_cmd,
        up_cmd,
        wake_cmd,
    )

    return app


@pytest.fixture
def tmp_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect all Silo paths to a temp directory."""
    config_dir = tmp_path / ".silo"
    config_dir.mkdir()

    registry_path = config_dir / "registry.json"
    config_file = config_dir / "config.yaml"
    logs_dir = config_dir / "logs"
    pids_dir = config_dir / "pids"

    # Patch at source
    monkeypatch.setattr("silo.config.paths.CONFIG_DIR", config_dir)
    monkeypatch.setattr("silo.config.paths.CONFIG_FILE", config_file)
    monkeypatch.setattr("silo.config.paths.REGISTRY_PATH", registry_path)
    monkeypatch.setattr("silo.config.paths.LOGS_DIR", logs_dir)
    monkeypatch.setattr("silo.config.paths.PIDS_DIR", pids_dir)

    # Patch at all import sites
    monkeypatch.setattr("silo.registry.store.REGISTRY_PATH", registry_path)
    monkeypatch.setattr("silo.cli.init_cmd.CONFIG_FILE", config_file)
    monkeypatch.setattr("silo.config.loader.CONFIG_FILE", config_file)

    return config_dir


@pytest.fixture
def mock_hf(monkeypatch: pytest.MonkeyPatch):
    """Mock huggingface_hub functions."""
    mock_info = MagicMock()
    mock_info.id = "test-org/test-model"
    mock_info.author = "test-org"
    mock_info.tags = ["mlx", "text-generation"]
    mock_info.downloads = 1000
    mock_info.likes = 50
    mock_info.pipeline_tag = "text-generation"
    mock_info.library_name = "mlx"
    mock_info.siblings = [
        MagicMock(rfilename="config.json"),
        MagicMock(rfilename="weights.safetensors"),
    ]

    monkeypatch.setattr("silo.download.hf.hf_model_info", lambda repo_id: mock_info)
    monkeypatch.setattr(
        "silo.download.hf.snapshot_download",
        lambda **kw: str(Path("/tmp/mock-models") / kw["repo_id"]),
    )

    return mock_info
