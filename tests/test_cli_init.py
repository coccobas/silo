"""Tests for init CLI command."""


class TestInitCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["init", "--help"])
        assert result.exit_code == 0
        assert "Generate" in result.output

    def test_init_creates_config(self, cli_runner, cli_app, tmp_config_dir):
        result = cli_runner.invoke(cli_app, ["init"])
        assert result.exit_code == 0
        config_file = tmp_config_dir / "config.yaml"
        assert config_file.exists()
        content = config_file.read_text()
        assert "models:" in content

    def test_init_no_overwrite(self, cli_runner, cli_app, tmp_config_dir):
        (tmp_config_dir / "config.yaml").write_text("existing")
        result = cli_runner.invoke(cli_app, ["init"])
        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_init_force_overwrite(self, cli_runner, cli_app, tmp_config_dir):
        (tmp_config_dir / "config.yaml").write_text("existing")
        result = cli_runner.invoke(cli_app, ["init", "--force"])
        assert result.exit_code == 0
        content = (tmp_config_dir / "config.yaml").read_text()
        assert "models:" in content
