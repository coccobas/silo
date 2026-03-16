"""Tests for run CLI command."""

from unittest.mock import MagicMock, patch

from silo.registry.models import ModelFormat, RegistryEntry
from silo.registry.store import Registry


class TestRunCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["run", "--help"])
        assert result.exit_code == 0
        assert "one-shot inference" in result.output

    def test_run_no_mlx(self, cli_runner, cli_app, tmp_config_dir, mock_hf):
        """Test that run fails gracefully when mlx-lm is not installed."""
        entry = RegistryEntry(
            repo_id="org/model", format=ModelFormat.MLX, local_path="/tmp/m"
        )
        Registry().add(entry).save(tmp_config_dir / "registry.json")

        with patch.dict("sys.modules", {"mlx_lm": None}):
            result = cli_runner.invoke(
                cli_app, ["run", "org/model", "Hello", "--no-stream"]
            )

        assert result.exit_code == 1

    def test_run_standard_model_warns(self, cli_runner, cli_app, tmp_config_dir):
        """Test that standard format models get a conversion suggestion."""
        mock_info_fn = MagicMock(return_value={
            "id": "org/standard",
            "siblings": [{"rfilename": "model.safetensors"}],
        })

        with patch("silo.download.hf.hf_model_info") as mock_hf_info:
            mock_hf_info.return_value = MagicMock(
                id="org/standard",
                siblings=[MagicMock(rfilename="model.safetensors")],
                author="org",
                tags=[],
                downloads=100,
                likes=5,
                pipeline_tag="text-generation",
            )
            with patch(
                "silo.registry.detector.detect_model_format",
                return_value=ModelFormat.STANDARD,
            ):
                result = cli_runner.invoke(
                    cli_app, ["run", "org/standard", "Hello"]
                )

        assert result.exit_code == 1
        assert "convert" in result.output.lower()
