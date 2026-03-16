"""Tests for doctor checks."""

from silo.doctor.checks import (
    CheckStatus,
    check_apple_silicon,
    check_ffmpeg,
    check_huggingface_hub,
    check_memory,
    check_python,
    check_registry,
    run_all_checks,
)


class TestChecks:
    def test_python_check(self):
        result = check_python()
        assert result.name == "Python"
        assert result.status == CheckStatus.OK

    def test_apple_silicon_check(self):
        result = check_apple_silicon()
        assert result.name == "Apple Silicon"
        # Status depends on actual hardware

    def test_huggingface_hub_check(self):
        result = check_huggingface_hub()
        assert result.name == "huggingface-hub"
        assert result.status == CheckStatus.OK  # installed as dependency

    def test_ffmpeg_check(self):
        result = check_ffmpeg()
        assert result.name == "ffmpeg"

    def test_memory_check(self):
        result = check_memory()
        assert result.name == "Memory"

    def test_registry_check_no_file(self, tmp_config_dir):
        result = check_registry()
        assert result.status == CheckStatus.OK

    def test_run_all_checks(self):
        results = run_all_checks()
        assert len(results) == 8
        assert all(hasattr(r, "status") for r in results)
