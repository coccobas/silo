"""Tests for PID file management."""

from __future__ import annotations

import os

import json

from silo.process.pid import (
    PidEntry,
    is_running,
    list_pid_entries,
    list_pids,
    read_pid,
    read_pid_entry,
    remove_pid,
    write_pid,
)


class TestWritePid:
    def test_write_creates_file(self, tmp_path):
        pid_file = write_pid("test", 12345, pids_dir=tmp_path)
        assert pid_file.exists()
        data = json.loads(pid_file.read_text())
        assert data["pid"] == 12345

    def test_write_with_metadata(self, tmp_path):
        write_pid("test", 99, port=8800, repo_id="org/model", runtime="mlx", pids_dir=tmp_path)
        entry = read_pid_entry("test", pids_dir=tmp_path)
        assert entry is not None
        assert entry.pid == 99
        assert entry.port == 8800
        assert entry.repo_id == "org/model"
        assert entry.runtime == "mlx"

    def test_write_overwrites(self, tmp_path):
        write_pid("test", 111, pids_dir=tmp_path)
        write_pid("test", 222, pids_dir=tmp_path)
        assert read_pid("test", pids_dir=tmp_path) == 222


class TestReadPid:
    def test_read_existing(self, tmp_path):
        write_pid("test", 12345, pids_dir=tmp_path)
        assert read_pid("test", pids_dir=tmp_path) == 12345

    def test_read_nonexistent(self, tmp_path):
        assert read_pid("nonexistent", pids_dir=tmp_path) is None

    def test_read_invalid_content(self, tmp_path):
        (tmp_path / "bad.pid").write_text("not a number")
        assert read_pid("bad", pids_dir=tmp_path) is None

    def test_read_legacy_plain_text(self, tmp_path):
        """Legacy PID files with just a number should still work."""
        (tmp_path / "legacy.pid").write_text("42")
        assert read_pid("legacy", pids_dir=tmp_path) == 42
        entry = read_pid_entry("legacy", pids_dir=tmp_path)
        assert entry is not None
        assert entry.pid == 42
        assert entry.port == 0
        assert entry.repo_id == ""


class TestRemovePid:
    def test_remove_existing(self, tmp_path):
        write_pid("test", 12345, pids_dir=tmp_path)
        remove_pid("test", pids_dir=tmp_path)
        assert not (tmp_path / "test.pid").exists()

    def test_remove_nonexistent(self, tmp_path):
        # Should not raise
        remove_pid("nonexistent", pids_dir=tmp_path)


class TestIsRunning:
    def test_current_process(self):
        assert is_running(os.getpid()) is True

    def test_nonexistent_process(self):
        assert is_running(99999999) is False


class TestListPids:
    def test_empty(self, tmp_path):
        assert list_pids(pids_dir=tmp_path) == {}

    def test_multiple(self, tmp_path):
        write_pid("model-a", 111, pids_dir=tmp_path)
        write_pid("model-b", 222, pids_dir=tmp_path)
        result = list_pids(pids_dir=tmp_path)
        assert result == {"model-a": 111, "model-b": 222}

    def test_nonexistent_dir(self, tmp_path):
        assert list_pids(pids_dir=tmp_path / "nope") == {}


class TestListPidEntries:
    def test_empty(self, tmp_path):
        assert list_pid_entries(pids_dir=tmp_path) == {}

    def test_entries_have_metadata(self, tmp_path):
        write_pid("a", 1, port=8800, repo_id="org/a", pids_dir=tmp_path)
        write_pid("b", 2, port=8801, repo_id="org/b", runtime="llamacpp", pids_dir=tmp_path)
        entries = list_pid_entries(pids_dir=tmp_path)
        assert entries["a"].port == 8800
        assert entries["a"].repo_id == "org/a"
        assert entries["b"].runtime == "llamacpp"
