"""Tests for serve CLI command."""

from unittest.mock import MagicMock, patch

from silo.registry.models import ModelFormat, RegistryEntry
from silo.registry.store import Registry


class TestServeCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Serve a model" in result.output
        assert "--host" in result.output
        assert "--port" in result.output

    def test_serve_registered_model(self, cli_runner, cli_app, tmp_config_dir):
        entry = RegistryEntry(
            repo_id="org/model", format=ModelFormat.MLX, local_path="/tmp/m"
        )
        Registry().add(entry).save(tmp_config_dir / "registry.json")

        mock_backend = MagicMock()
        mock_backend.health.return_value = {"status": "ok", "model": "org/model", "backend": "mock"}

        with patch("silo.backends.factory.resolve_backend", return_value=mock_backend), \
             patch("uvicorn.run") as mock_run:
            result = cli_runner.invoke(cli_app, ["serve", "org/model"])

        assert result.exit_code == 0
        assert "Serving" in result.output
        mock_run.assert_called_once()

    def test_serve_env_overrides(self, cli_runner, cli_app, tmp_config_dir, monkeypatch):
        entry = RegistryEntry(
            repo_id="org/model", format=ModelFormat.MLX, local_path="/tmp/m"
        )
        Registry().add(entry).save(tmp_config_dir / "registry.json")

        monkeypatch.setenv("SILO_HOST", "0.0.0.0")
        monkeypatch.setenv("SILO_PORT", "9999")

        mock_backend = MagicMock()

        with patch("silo.backends.factory.resolve_backend", return_value=mock_backend), \
             patch("uvicorn.run") as mock_run:
            result = cli_runner.invoke(cli_app, ["serve", "org/model"])

        assert result.exit_code == 0
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["host"] == "0.0.0.0"
        assert call_kwargs.kwargs["port"] == 9999

    def test_serve_standard_without_quantize(self, cli_runner, cli_app, tmp_config_dir):
        mock_info = {
            "id": "org/standard",
            "siblings": [{"rfilename": "model.safetensors"}],
            "author": "org",
            "tags": [],
            "downloads": 100,
            "likes": 5,
            "pipeline_tag": "text-generation",
            "library_name": None,
        }
        with patch("silo.download.hf.get_model_info", return_value=mock_info), \
             patch(
                "silo.registry.detector.detect_model_format",
                return_value=ModelFormat.STANDARD,
             ):
            result = cli_runner.invoke(cli_app, ["serve", "org/standard"])

        assert result.exit_code == 1
        assert "convert" in result.output.lower()

    def test_serve_with_custom_name(self, cli_runner, cli_app, tmp_config_dir):
        entry = RegistryEntry(
            repo_id="org/model", format=ModelFormat.MLX, local_path="/tmp/m"
        )
        Registry().add(entry).save(tmp_config_dir / "registry.json")

        mock_backend = MagicMock()

        with patch("silo.backends.factory.resolve_backend", return_value=mock_backend), \
             patch("uvicorn.run"):
            result = cli_runner.invoke(
                cli_app, ["serve", "org/model", "--name", "llama"]
            )

        assert result.exit_code == 0
        assert "llama" in result.output

    def test_serve_fetch_error(self, cli_runner, cli_app, tmp_config_dir):
        with patch(
            "silo.download.hf.get_model_info",
            side_effect=Exception("Network error"),
        ):
            result = cli_runner.invoke(cli_app, ["serve", "bad/model"])

        assert result.exit_code == 1
        assert "Could not fetch model info" in result.output
