"""Tests for doctor CLI command."""


class TestDoctorCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "Diagnose" in result.output

    def test_doctor_runs(self, cli_runner, cli_app, tmp_config_dir):
        result = cli_runner.invoke(cli_app, ["doctor"])
        assert "Silo Doctor" in result.output
        assert "Python" in result.output
