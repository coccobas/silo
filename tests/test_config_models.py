"""Tests for config Pydantic models."""

import pytest
from pydantic import ValidationError

from silo.config.models import AppConfig, ModelConfig, RestartPolicy


class TestModelConfig:
    def test_defaults(self):
        config = ModelConfig(name="test", repo="org/model")
        assert config.host == "127.0.0.1"
        assert config.port == 8800
        assert config.warmup is False
        assert config.restart == RestartPolicy.ON_FAILURE
        assert config.timeout == 120
        assert config.backend is None
        assert config.quantize is None
        assert config.output is None

    def test_custom_values(self):
        config = ModelConfig(
            name="llama",
            repo="mlx-community/Llama-3.2-1B-4bit",
            host="0.0.0.0",
            port=9000,
            warmup=True,
            restart=RestartPolicy.ALWAYS,
            timeout=60,
        )
        assert config.name == "llama"
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.warmup is True

    def test_frozen(self):
        config = ModelConfig(name="test", repo="org/model")
        with pytest.raises(ValidationError):
            config.name = "changed"  # type: ignore[misc]

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            ModelConfig(name="test")  # type: ignore[call-arg]


class TestAppConfig:
    def test_empty(self):
        config = AppConfig()
        assert config.models == []

    def test_with_models(self):
        config = AppConfig(
            models=[ModelConfig(name="a", repo="org/a"), ModelConfig(name="b", repo="org/b")]
        )
        assert len(config.models) == 2

    def test_frozen(self):
        config = AppConfig()
        with pytest.raises(ValidationError):
            config.models = []  # type: ignore[misc]
