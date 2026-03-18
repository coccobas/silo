"""Tests for process management CLI commands (up, down, ps, logs)."""

from __future__ import annotations

from unittest.mock import patch

from silo.config.models import AppConfig, ModelConfig
from silo.process.manager import SpawnResult
from silo.process.pid import PidEntry


def _make_config(*models):
    """Helper to create an AppConfig with model configs."""
    return AppConfig(models=list(models))


def _model(name="test", repo="org/model", port=8800):
    return ModelConfig(name=name, repo=repo, port=port)


class TestUpCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["up", "--help"])
        assert result.exit_code == 0
        assert "Bring up" in result.output

    def test_no_config(self, cli_runner, cli_app, tmp_config_dir):
        with patch("silo.config.loader.load_config", return_value=AppConfig()):
            result = cli_runner.invoke(cli_app, ["up"])
        assert result.exit_code == 1
        assert "No models configured" in result.output

    def test_model_not_found(self, cli_runner, cli_app, tmp_config_dir):
        config = _make_config(_model("llama"))
        with patch("silo.config.loader.load_config", return_value=config):
            result = cli_runner.invoke(cli_app, ["up", "nonexistent"])
        assert result.exit_code == 1
        assert "not found in config" in result.output

    def test_already_running(self, cli_runner, cli_app, tmp_config_dir):
        config = _make_config(_model("llama"))
        with patch("silo.config.loader.load_config", return_value=config), \
             patch("silo.process.pid.read_pid", return_value=123), \
             patch("silo.process.pid.is_running", return_value=True):
            result = cli_runner.invoke(cli_app, ["up", "llama"])
        assert result.exit_code == 0
        assert "already running" in result.output

    def test_start_model(self, cli_runner, cli_app, tmp_config_dir):
        config = _make_config(_model("llama"))
        spawn_result = SpawnResult(pid=42, instance_id="test-uuid")
        with patch("silo.config.loader.load_config", return_value=config), \
             patch("silo.process.pid.read_pid", return_value=None), \
             patch("silo.process.manager.spawn_model", return_value=spawn_result) as mock_spawn:
            result = cli_runner.invoke(cli_app, ["up", "llama"])
        assert result.exit_code == 0
        assert "Starting llama" in result.output
        assert "42" in result.output
        mock_spawn.assert_called_once()


class TestDownCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["down", "--help"])
        assert result.exit_code == 0
        assert "Stop" in result.output

    def test_stop_specific(self, cli_runner, cli_app, tmp_config_dir):
        entry = PidEntry(pid=1, instance_id="uuid-1")
        with patch("silo.config.loader.load_config", return_value=AppConfig()), \
             patch("silo.process.pid.read_pid_entry", return_value=entry), \
             patch("silo.process.manager.stop_model", return_value=True):
            result = cli_runner.invoke(cli_app, ["down", "llama"])
        assert result.exit_code == 0
        assert "Stopped llama" in result.output

    def test_stop_not_running(self, cli_runner, cli_app, tmp_config_dir):
        with patch("silo.config.loader.load_config", return_value=AppConfig()), \
             patch("silo.process.pid.read_pid_entry", return_value=None), \
             patch("silo.process.manager.stop_model", return_value=False):
            result = cli_runner.invoke(cli_app, ["down", "llama"])
        assert result.exit_code == 0
        assert "not running" in result.output

    def test_stop_all_none(self, cli_runner, cli_app, tmp_config_dir):
        with patch("silo.config.loader.load_config", return_value=AppConfig()), \
             patch("silo.process.pid.list_pid_entries", return_value={}):
            result = cli_runner.invoke(cli_app, ["down"])
        assert result.exit_code == 0
        assert "No running" in result.output

    def test_stop_all(self, cli_runner, cli_app, tmp_config_dir):
        entries = {
            "a": PidEntry(pid=1, instance_id="uuid-a"),
            "b": PidEntry(pid=2, instance_id="uuid-b"),
        }
        with patch("silo.config.loader.load_config", return_value=AppConfig()), \
             patch("silo.process.pid.list_pid_entries", return_value=entries), \
             patch("silo.process.manager.stop_model", return_value=True):
            result = cli_runner.invoke(cli_app, ["down"])
        assert result.exit_code == 0
        assert "Stopped a" in result.output
        assert "Stopped b" in result.output


class TestPsCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["ps", "--help"])
        assert result.exit_code == 0
        assert "status" in result.output.lower()

    def test_no_config_no_running(self, cli_runner, cli_app, tmp_config_dir):
        with patch("silo.config.loader.load_config", return_value=AppConfig()), \
             patch("silo.process.manager.list_running", return_value=[]):
            result = cli_runner.invoke(cli_app, ["ps"])
        assert result.exit_code == 0
        assert "No running" in result.output

    def test_with_config(self, cli_runner, cli_app, tmp_config_dir):
        from silo.process.manager import ProcessInfo

        config = _make_config(_model("llama", port=8800))
        info = ProcessInfo(name="llama", pid=42, port=8800, repo_id="org/model", status="running")

        with patch("silo.config.loader.load_config", return_value=config), \
             patch("silo.process.manager.get_status", return_value=info):
            result = cli_runner.invoke(cli_app, ["ps"])
        assert result.exit_code == 0
        assert "llama" in result.output


class TestLogsCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["logs", "--help"])
        assert result.exit_code == 0
        assert "logs" in result.output.lower()

    def test_no_logs(self, cli_runner, cli_app, tmp_config_dir):
        result = cli_runner.invoke(cli_app, ["logs", "nonexistent"])
        assert result.exit_code == 1
        assert "No logs found" in result.output

    def test_tail_logs(self, cli_runner, cli_app, tmp_config_dir):
        log_file = tmp_config_dir / "logs" / "test.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("line1\nline2\nline3\n")

        with patch("silo.cli.logs_cmd.LOGS_DIR", tmp_config_dir / "logs"):
            result = cli_runner.invoke(cli_app, ["logs", "test", "--tail", "2"])
        assert result.exit_code == 0
        assert "line2" in result.output
        assert "line3" in result.output
