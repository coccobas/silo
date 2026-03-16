"""Tests for config loader."""

from pathlib import Path

import pytest

from silo.config.loader import load_config
from silo.config.models import AppConfig


class TestLoadConfig:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        config = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(config, AppConfig)
        assert config.models == []

    def test_empty_file(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        config = load_config(config_file)
        assert config.models == []

    def test_valid_config(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
models:
  - name: llama
    repo: mlx-community/Llama-3.2-1B-4bit
    port: 8800
  - name: whisper
    repo: mlx-community/whisper-large-v3-turbo
    port: 8801
"""
        )
        config = load_config(config_file)
        assert len(config.models) == 2
        assert config.models[0].name == "llama"
        assert config.models[1].port == 8801

    def test_env_override_host(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
models:
  - name: test
    repo: org/model
    host: 127.0.0.1
"""
        )
        monkeypatch.setenv("SILO_HOST", "0.0.0.0")
        config = load_config(config_file)
        assert config.models[0].host == "0.0.0.0"

    def test_env_override_port(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
models:
  - name: test
    repo: org/model
    port: 8800
"""
        )
        monkeypatch.setenv("SILO_PORT", "9000")
        config = load_config(config_file)
        assert config.models[0].port == 9000
