"""Tests for models CLI commands."""

from unittest.mock import MagicMock, patch

from silo.registry.models import ModelFormat, RegistryEntry
from silo.registry.store import Registry


class TestModelsList:
    def test_empty_list(self, cli_runner, cli_app, tmp_config_dir):
        result = cli_runner.invoke(cli_app, ["models", "list"])
        assert result.exit_code == 0
        assert "No models registered" in result.output

    def test_list_with_models(self, cli_runner, cli_app, tmp_config_dir):
        entry = RegistryEntry(repo_id="org/model", format=ModelFormat.MLX, local_path="/tmp/m")
        Registry().add(entry).save(tmp_config_dir / "registry.json")

        result = cli_runner.invoke(cli_app, ["models", "list"])
        assert result.exit_code == 0
        assert "org/model" in result.output


class TestModelsInfo:
    def test_info_local(self, cli_runner, cli_app, tmp_config_dir):
        entry = RegistryEntry(repo_id="org/model", format=ModelFormat.MLX)
        Registry().add(entry).save(tmp_config_dir / "registry.json")

        result = cli_runner.invoke(cli_app, ["models", "info", "org/model"])
        assert result.exit_code == 0
        assert "org/model" in result.output

    def test_info_remote(self, cli_runner, cli_app, tmp_config_dir, mock_hf):
        result = cli_runner.invoke(cli_app, ["models", "info", "test-org/test-model"])
        assert result.exit_code == 0
        assert "test-org/test-model" in result.output


class TestModelsRm:
    def test_rm_existing(self, cli_runner, cli_app, tmp_config_dir):
        entry = RegistryEntry(repo_id="org/model", format=ModelFormat.MLX)
        Registry().add(entry).save(tmp_config_dir / "registry.json")

        result = cli_runner.invoke(cli_app, ["models", "rm", "org/model"])
        assert result.exit_code == 0
        assert "Removed" in result.output

        reg = Registry.load(tmp_config_dir / "registry.json")
        assert reg.get("org/model") is None

    def test_rm_nonexistent(self, cli_runner, cli_app, tmp_config_dir):
        result = cli_runner.invoke(cli_app, ["models", "rm", "nonexistent"])
        assert result.exit_code == 1


class TestModelsSearch:
    def test_search(self, cli_runner, cli_app):
        mock_model = MagicMock()
        mock_model.id = "mlx-community/test"
        mock_model.author = "mlx-community"
        mock_model.downloads = 500
        mock_model.likes = 10
        mock_model.pipeline_tag = "text-generation"
        mock_model.tags = ["mlx"]

        with patch("huggingface_hub.list_models", return_value=[mock_model]):
            result = cli_runner.invoke(cli_app, ["models", "search", "test"])

        assert result.exit_code == 0
        assert "mlx-community/test" in result.output
