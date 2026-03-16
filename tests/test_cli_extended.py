"""Extended CLI tests for edge cases and uncovered paths."""

from unittest.mock import MagicMock, patch

from silo.registry.models import ModelFormat, RegistryEntry
from silo.registry.store import Registry


class TestAppVersion:
    def test_version_flag(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["--version"])
        assert result.exit_code == 0
        assert "silo 0.9.0" in result.output


class TestAppMain:
    def test_main_entry(self):
        from silo.cli.app import main

        with patch("silo.cli.app.app") as mock_app:
            main()
            mock_app.assert_called_once()


class TestModelsInfoExtended:
    def test_info_with_size_and_tags(self, cli_runner, cli_app, tmp_config_dir):
        entry = RegistryEntry(
            repo_id="org/model",
            format=ModelFormat.MLX,
            size_bytes=1024 * 1024 * 500,
            tags=["converted", "quantized-q4"],
        )
        Registry().add(entry).save(tmp_config_dir / "registry.json")

        result = cli_runner.invoke(cli_app, ["models", "info", "org/model"])
        assert result.exit_code == 0
        assert "500.0 MB" in result.output
        assert "converted" in result.output

    def test_info_remote_error(self, cli_runner, cli_app, tmp_config_dir):
        with patch(
            "silo.download.hf.hf_model_info",
            side_effect=Exception("Network error"),
        ):
            result = cli_runner.invoke(cli_app, ["models", "info", "bad/model"])
        assert result.exit_code == 1
        assert "Error fetching model info" in result.output


class TestModelsRmExtended:
    def test_rm_purge(self, cli_runner, cli_app, tmp_config_dir, tmp_path):
        model_dir = tmp_path / "model_files"
        model_dir.mkdir()
        (model_dir / "weights.safetensors").write_text("fake")

        entry = RegistryEntry(
            repo_id="org/model",
            format=ModelFormat.MLX,
            local_path=str(model_dir),
        )
        Registry().add(entry).save(tmp_config_dir / "registry.json")

        result = cli_runner.invoke(cli_app, ["models", "rm", "org/model", "--purge"], input="y\n")
        assert result.exit_code == 0
        assert "Removed" in result.output


class TestModelsSearchExtended:
    def test_search_no_results(self, cli_runner, cli_app):
        with patch("huggingface_hub.list_models", return_value=[]):
            result = cli_runner.invoke(cli_app, ["models", "search", "nonexistent"])
        assert result.exit_code == 0
        assert "No models found" in result.output


class TestConvertExtended:
    def test_convert_with_quantize(self, cli_runner, cli_app, tmp_config_dir):
        with patch(
            "silo.convert.mlx.convert_model",
            return_value="/tmp/out",
        ):
            result = cli_runner.invoke(
                cli_app, ["convert", "org/model", "-q", "q4", "-o", "/tmp/out"]
            )
        assert result.exit_code == 0
        assert "Quantization: q4" in result.output

    def test_convert_generic_error(self, cli_runner, cli_app, tmp_config_dir):
        with patch(
            "silo.convert.mlx.convert_model",
            side_effect=RuntimeError("Something broke"),
        ):
            result = cli_runner.invoke(
                cli_app, ["convert", "org/model", "-o", "/tmp/out"]
            )
        assert result.exit_code == 1
        assert "Conversion failed" in result.output


class TestDoctorExtended:
    def test_doctor_with_failure(self, cli_runner, cli_app, tmp_config_dir):
        from silo.doctor.checks import CheckResult, CheckStatus

        failing_checks = [
            CheckResult("Python", CheckStatus.FAIL, "3.10 (need 3.12+)"),
            CheckResult("MLX", CheckStatus.OK, "v0.30"),
        ]
        with patch("silo.doctor.checks.run_all_checks", return_value=failing_checks):
            result = cli_runner.invoke(cli_app, ["doctor"])
        assert result.exit_code == 1
        assert "Some checks failed" in result.output


class TestRunExtended:
    def test_run_auto_download(self, cli_runner, cli_app, tmp_config_dir, mock_hf):
        """Test auto-download + inference with mocked backend."""
        mock_backend = MagicMock()
        mock_backend.chat.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
        }

        with patch("silo.backends.mlx_lm.MlxLmBackend", return_value=mock_backend):
            result = cli_runner.invoke(
                cli_app,
                ["run", "test-org/test-model", "Hi there", "--no-stream"],
            )

        assert result.exit_code == 0
        assert "Hello!" in result.output

    def test_run_streaming(self, cli_runner, cli_app, tmp_config_dir):
        entry = RegistryEntry(
            repo_id="org/model", format=ModelFormat.MLX, local_path="/tmp/m"
        )
        Registry().add(entry).save(tmp_config_dir / "registry.json")

        chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ]
        mock_backend = MagicMock()
        mock_backend.chat.return_value = iter(chunks)

        with patch("silo.backends.mlx_lm.MlxLmBackend", return_value=mock_backend):
            result = cli_runner.invoke(
                cli_app,
                ["run", "org/model", "Hi", "--stream"],
            )

        assert result.exit_code == 0
        assert "Hello" in result.output
        assert "world" in result.output

    def test_run_fetch_error(self, cli_runner, cli_app, tmp_config_dir):
        with patch(
            "silo.download.hf.hf_model_info",
            side_effect=Exception("Network error"),
        ):
            result = cli_runner.invoke(
                cli_app, ["run", "bad/model", "Hello"]
            )

        assert result.exit_code == 1
        assert "Could not fetch model info" in result.output


class TestRegistryStoreError:
    def test_save_cleanup_on_error(self, tmp_path):
        """Test that temp file is cleaned up on write error."""
        from silo.registry.models import RegistryEntry
        from silo.registry.store import Registry

        entry = RegistryEntry(repo_id="org/model", format=ModelFormat.MLX)
        reg = Registry().add(entry)

        # Make the target path a directory to cause os.rename to fail
        target = tmp_path / "registry.json"
        target.mkdir()

        with patch("silo.registry.store.ensure_dirs"):
            try:
                reg.save(target)
            except (OSError, IsADirectoryError):
                pass

        # Temp file should be cleaned up
        import glob

        tmp_files = glob.glob(str(tmp_path / "*.json.tmp"))
        assert len(tmp_files) == 0
