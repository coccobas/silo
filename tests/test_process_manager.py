"""Tests for process manager."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from silo.process.manager import ProcessInfo, get_status, list_running, spawn_model, stop_model
from silo.process.pid import read_pid, write_pid


class TestSpawnModel:
    def test_spawn_creates_pid(self, tmp_path):
        pids_dir = tmp_path / "pids"
        pids_dir.mkdir()
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        mock_proc = MagicMock()
        mock_proc.pid = 42

        with patch("silo.process.manager.subprocess.Popen", return_value=mock_proc):
            pid = spawn_model(
                "test-model", "org/model",
                pids_dir=pids_dir, logs_dir=logs_dir,
            )

        assert pid == 42
        assert read_pid("test-model", pids_dir=pids_dir) == 42

    def test_spawn_with_options(self, tmp_path):
        pids_dir = tmp_path / "pids"
        pids_dir.mkdir()
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        mock_proc = MagicMock()
        mock_proc.pid = 43

        with patch("silo.process.manager.subprocess.Popen", return_value=mock_proc) as mock_popen:
            spawn_model(
                "test-model", "org/model",
                host="0.0.0.0", port=9999,
                quantize="q4", output="/tmp/out",
                pids_dir=pids_dir, logs_dir=logs_dir,
            )

        cmd = mock_popen.call_args[0][0]
        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        assert "--port" in cmd
        assert "9999" in cmd
        assert "--quantize" in cmd
        assert "q4" in cmd
        assert "--output" in cmd
        assert "/tmp/out" in cmd


class TestStopModel:
    def test_stop_not_found(self, tmp_path):
        result = stop_model("nonexistent", pids_dir=tmp_path)
        assert result is False

    def test_stop_already_dead(self, tmp_path):
        write_pid("test", 99999999, pids_dir=tmp_path)
        result = stop_model("test", pids_dir=tmp_path)
        assert result is True
        assert read_pid("test", pids_dir=tmp_path) is None

    def test_stop_running_process(self, tmp_path):
        write_pid("test", os.getpid(), pids_dir=tmp_path)

        with patch("silo.process.manager.is_running") as mock_running, \
             patch("os.kill") as mock_kill:
            # First call: running. After SIGTERM: not running.
            mock_running.side_effect = [True, False]
            result = stop_model("test", pids_dir=tmp_path)

        assert result is True
        mock_kill.assert_called_once()


class TestGetStatus:
    def test_no_pid_file(self, tmp_path):
        info = get_status("test", port=8800, repo_id="org/m", pids_dir=tmp_path)
        assert info.status == "stopped"
        assert info.pid == 0

    def test_running(self, tmp_path):
        write_pid("test", os.getpid(), pids_dir=tmp_path)
        info = get_status("test", port=8800, repo_id="org/m", pids_dir=tmp_path)
        assert info.status == "running"
        assert info.pid == os.getpid()

    def test_stale_pid(self, tmp_path):
        write_pid("test", 99999999, pids_dir=tmp_path)
        info = get_status("test", port=8800, repo_id="org/m", pids_dir=tmp_path)
        assert info.status == "stopped"
        assert info.pid == 0


class TestListRunning:
    def test_empty(self, tmp_path):
        assert list_running(pids_dir=tmp_path) == []

    def test_mixed(self, tmp_path):
        write_pid("running", os.getpid(), pids_dir=tmp_path)
        write_pid("dead", 99999999, pids_dir=tmp_path)

        result = list_running(pids_dir=tmp_path)
        names = {p.name for p in result}
        assert "running" in names
        assert "dead" in names

        running = [p for p in result if p.name == "running"][0]
        assert running.status == "running"

        dead = [p for p in result if p.name == "dead"][0]
        assert dead.status == "stopped"
