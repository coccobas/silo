"""Tests for the wake CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def flow_file(tmp_config_dir):
    """Create a valid flow file for testing."""
    flows_dir = tmp_config_dir / "flows"
    flows_dir.mkdir()
    flow_path = flows_dir / "test-flow.yaml"
    flow_path.write_text(
        "name: test-flow\n"
        "steps:\n"
        "  - id: greet\n"
        "    type: fs.write\n"
        "    input: {content: hello, path: out.txt}\n"
    )
    return flow_path


class TestWakeCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["wake", "--help"])
        assert result.exit_code == 0
        assert "flow" in result.output.lower()

    def test_missing_flow_flag(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["wake"])
        assert result.exit_code != 0

    def test_nonexistent_flow(
        self, cli_runner, cli_app, tmp_config_dir
    ):
        result = cli_runner.invoke(
            cli_app, ["wake", "--flow", "does-not-exist"]
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_valid_flow_starts_listener(
        self, cli_runner, cli_app, flow_file
    ):
        mock_listener = MagicMock()
        mock_cls = MagicMock(return_value=mock_listener)

        with patch(
            "silo.wake.listener.WakeWordListener", mock_cls
        ):
            cli_runner.invoke(
                cli_app,
                ["wake", "--flow", "test-flow", "--once"],
            )

        mock_cls.assert_called_once()
        mock_listener.run.assert_called_once()

    def test_custom_word_and_threshold(
        self, cli_runner, cli_app, flow_file
    ):
        mock_listener = MagicMock()
        mock_cls = MagicMock(return_value=mock_listener)

        with patch(
            "silo.wake.listener.WakeWordListener", mock_cls
        ):
            cli_runner.invoke(
                cli_app,
                [
                    "wake",
                    "--flow", "test-flow",
                    "--word", "alexa",
                    "--threshold", "0.8",
                    "--once",
                ],
            )

        config = mock_cls.call_args[1]["config"]
        assert config.wake_word == "alexa"
        assert config.threshold == 0.8

    def test_onnx_model_path(
        self, cli_runner, cli_app, flow_file
    ):
        mock_listener = MagicMock()
        mock_cls = MagicMock(return_value=mock_listener)

        with patch(
            "silo.wake.listener.WakeWordListener", mock_cls
        ):
            cli_runner.invoke(
                cli_app,
                [
                    "wake",
                    "--flow", "test-flow",
                    "--word", "/path/to/custom.onnx",
                    "--once",
                ],
            )

        config = mock_cls.call_args[1]["config"]
        assert config.model_path == "/path/to/custom.onnx"

    def test_import_error_handled(
        self, cli_runner, cli_app, flow_file
    ):
        mock_listener = MagicMock()
        mock_listener.run.side_effect = ImportError(
            "openwakeword not installed"
        )
        mock_cls = MagicMock(return_value=mock_listener)

        with patch(
            "silo.wake.listener.WakeWordListener", mock_cls
        ):
            result = cli_runner.invoke(
                cli_app,
                ["wake", "--flow", "test-flow", "--once"],
            )

        assert result.exit_code != 0

    def test_runtime_error_handled(
        self, cli_runner, cli_app, flow_file
    ):
        mock_listener = MagicMock()
        mock_listener.run.side_effect = RuntimeError(
            "Microphone access denied"
        )
        mock_cls = MagicMock(return_value=mock_listener)

        with patch(
            "silo.wake.listener.WakeWordListener", mock_cls
        ):
            result = cli_runner.invoke(
                cli_app,
                ["wake", "--flow", "test-flow", "--once"],
            )

        assert result.exit_code != 0
