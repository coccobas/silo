"""Tests for memory monitoring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from silo.process.memory import MemoryInfo, check_memory_pressure, get_memory_info


class TestMemoryInfo:
    def test_usage_percent(self):
        info = MemoryInfo(total_gb=16.0, available_gb=4.0, used_gb=12.0, pressure="normal")
        assert info.usage_percent == 75.0

    def test_usage_percent_zero_total(self):
        info = MemoryInfo(total_gb=0, available_gb=0, used_gb=0, pressure="unknown")
        assert info.usage_percent == 0.0

    def test_frozen(self):
        info = MemoryInfo(total_gb=16.0, available_gb=4.0, used_gb=12.0, pressure="normal")
        try:
            info.total_gb = 32.0  # type: ignore
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass


class TestGetMemoryInfo:
    def test_non_darwin(self):
        with patch("silo.process.memory.platform.system", return_value="Linux"):
            info = get_memory_info()
        assert info.pressure == "unknown"
        assert info.total_gb == 0

    def test_darwin_success(self):
        mock_sysctl = MagicMock()
        mock_sysctl.stdout = "17179869184"  # 16 GB

        mock_pressure = MagicMock()
        mock_pressure.stdout = "System-wide memory free percentage: 50%"

        mock_vmstat = MagicMock()
        mock_vmstat.stdout = (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free:                      100000.\n"
            "Pages inactive:                   50000.\n"
            "Pages active:                    200000.\n"
        )

        with patch("silo.process.memory.platform.system", return_value="Darwin"), \
             patch("silo.process.memory.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_sysctl, mock_pressure, mock_vmstat]
            info = get_memory_info()

        assert info.total_gb == 16.0
        assert info.pressure == "normal"
        assert info.available_gb > 0

    def test_darwin_critical_pressure(self):
        mock_sysctl = MagicMock()
        mock_sysctl.stdout = "17179869184"

        mock_pressure = MagicMock()
        mock_pressure.stdout = "The system is in a critical memory state"

        mock_vmstat = MagicMock()
        mock_vmstat.stdout = (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free:                      1000.\n"
            "Pages inactive:                  500.\n"
        )

        with patch("silo.process.memory.platform.system", return_value="Darwin"), \
             patch("silo.process.memory.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_sysctl, mock_pressure, mock_vmstat]
            info = get_memory_info()

        assert info.pressure == "critical"

    def test_darwin_error(self):
        import subprocess

        with patch("silo.process.memory.platform.system", return_value="Darwin"), \
             patch("silo.process.memory.subprocess.run",
                   side_effect=subprocess.SubprocessError()):
            info = get_memory_info()

        assert info.pressure == "unknown"


class TestCheckMemoryPressure:
    def test_below_threshold(self):
        info = MemoryInfo(total_gb=16.0, available_gb=8.0, used_gb=8.0, pressure="normal")
        with patch("silo.process.memory.get_memory_info", return_value=info):
            assert check_memory_pressure() is False

    def test_above_threshold(self):
        info = MemoryInfo(total_gb=16.0, available_gb=1.0, used_gb=15.0, pressure="warn")
        with patch("silo.process.memory.get_memory_info", return_value=info):
            assert check_memory_pressure() is True

    def test_custom_threshold(self):
        info = MemoryInfo(total_gb=16.0, available_gb=4.0, used_gb=12.0, pressure="normal")
        with patch("silo.process.memory.get_memory_info", return_value=info):
            assert check_memory_pressure(threshold_percent=70.0) is True
            assert check_memory_pressure(threshold_percent=80.0) is False
