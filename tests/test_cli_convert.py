"""Tests for convert CLI command."""

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestConvertCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "Convert a model" in result.output

    def test_convert_success(self, cli_runner, cli_app, tmp_config_dir):
        mock_convert = MagicMock()
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.convert = mock_convert

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            # Need to also patch the convert_model function to avoid re-import issues
            with patch(
                "silo.convert.mlx.convert_model",
                return_value=Path("/tmp/test-out"),
            ):
                result = cli_runner.invoke(
                    cli_app, ["convert", "org/model", "--output", "/tmp/test-out"]
                )

        assert result.exit_code == 0
        assert "Converted" in result.output or "test-out" in result.output

    def test_convert_no_mlx(self, cli_runner, cli_app, tmp_config_dir):
        with patch(
            "silo.convert.mlx.convert_model",
            side_effect=ImportError("mlx-lm is required"),
        ):
            result = cli_runner.invoke(
                cli_app, ["convert", "org/model", "--output", "/tmp/test-out"]
            )

        assert result.exit_code == 1
        assert "mlx-lm is required" in result.output
