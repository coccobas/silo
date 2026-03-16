"""Extended tests for doctor checks — covers edge cases and failure paths."""

from unittest.mock import MagicMock, patch

from silo.doctor.checks import (
    CheckStatus,
    check_apple_silicon,
    check_ffmpeg,
    check_huggingface_hub,
    check_memory,
    check_mlx,
    check_mlx_lm,
    check_python,
    check_registry,
)


class TestCheckPythonEdgeCases:
    def test_old_python(self):
        mock_version = MagicMock()
        mock_version.major = 3
        mock_version.minor = 10
        mock_version.micro = 0
        mock_version.__ge__ = lambda self, other: (3, 10) >= other
        with patch("silo.doctor.checks.sys") as mock_sys:
            mock_sys.version_info = mock_version
            result = check_python()
        assert result.status == CheckStatus.FAIL
        assert "need 3.12+" in result.message


class TestCheckAppleSiliconEdgeCases:
    def test_not_macos(self):
        with patch("silo.doctor.checks.platform") as mock_platform:
            mock_platform.machine.return_value = "x86_64"
            mock_platform.system.return_value = "Linux"
            result = check_apple_silicon()
        assert result.status == CheckStatus.FAIL
        assert "Not macOS" in result.message

    def test_macos_x86(self):
        with patch("silo.doctor.checks.platform") as mock_platform:
            mock_platform.machine.return_value = "x86_64"
            mock_platform.system.return_value = "Darwin"
            result = check_apple_silicon()
        assert result.status == CheckStatus.WARN
        assert "x86_64" in result.message


class TestCheckMlx:
    def test_mlx_installed(self):
        mock_mlx = MagicMock()
        mock_mlx.__version__ = "0.30.0"
        with patch.dict("sys.modules", {"mlx": mock_mlx}):
            result = check_mlx()
        assert result.status == CheckStatus.OK
        assert "0.30.0" in result.message


class TestCheckMlxLm:
    def test_mlx_lm_installed(self):
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.__version__ = "0.22.0"
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = check_mlx_lm()
        assert result.status == CheckStatus.OK
        assert "0.22.0" in result.message


class TestCheckHuggingfaceHub:
    def test_not_installed(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            result = check_huggingface_hub()
        assert result.status == CheckStatus.FAIL


class TestCheckFfmpeg:
    def test_not_found(self):
        with patch("silo.doctor.checks.shutil.which", return_value=None):
            result = check_ffmpeg()
        assert result.status == CheckStatus.WARN
        assert "Not found" in result.message


class TestCheckMemory:
    def test_detection_failure(self):
        with patch("subprocess.run", side_effect=Exception("fail")):
            result = check_memory()
        assert result.status == CheckStatus.WARN
        assert "Could not detect" in result.message


class TestCheckRegistry:
    def test_with_existing_models(self, tmp_config_dir):
        from silo.registry.models import ModelFormat, RegistryEntry
        from silo.registry.store import Registry

        entry = RegistryEntry(repo_id="org/model", format=ModelFormat.MLX)
        Registry().add(entry).save(tmp_config_dir / "registry.json")

        result = check_registry()
        assert result.status == CheckStatus.OK
        assert "1 model(s)" in result.message

    def test_corrupt_registry(self, tmp_config_dir):
        (tmp_config_dir / "registry.json").write_text("{invalid json!!!")
        result = check_registry()
        assert result.status == CheckStatus.FAIL
        assert "Corrupt" in result.message
