"""Tests for flow CLI commands."""

from __future__ import annotations

from unittest.mock import patch

from silo.flows.parser import FlowDefinition, FlowStep


class TestFlowListCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["flow", "list", "--help"])
        assert result.exit_code == 0

    def test_no_flows(self, cli_runner, cli_app, tmp_config_dir):
        result = cli_runner.invoke(cli_app, ["flow", "list"])
        assert result.exit_code == 0
        assert "No flows found" in result.output

    def test_with_flows(self, cli_runner, cli_app, tmp_config_dir):
        flows_dir = tmp_config_dir / "flows"
        flows_dir.mkdir()
        (flows_dir / "test.yaml").write_text("""
name: test-flow
description: A test flow
steps:
  - id: s1
    type: text.generate
""")
        result = cli_runner.invoke(cli_app, ["flow", "list"])
        assert result.exit_code == 0
        assert "test-flow" in result.output


class TestFlowRunCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["flow", "run", "--help"])
        assert result.exit_code == 0

    def test_flow_not_found(self, cli_runner, cli_app, tmp_config_dir):
        result = cli_runner.invoke(cli_app, ["flow", "run", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_run_flow_file(self, cli_runner, cli_app, tmp_config_dir, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.txt").write_text("hello")

        flow_file = tmp_path / "test-flow.yaml"
        flow_file.write_text("""
name: glob-test
steps:
  - id: glob
    type: fs.glob
    input: "*.txt"
""")
        result = cli_runner.invoke(cli_app, ["flow", "run", str(flow_file)])
        assert result.exit_code == 0
        assert "completed successfully" in result.output
