"""Tests for flow YAML parser."""

from __future__ import annotations

import pytest

from silo.flows.parser import FlowConfig, FlowDefinition, FlowStep, list_flows, parse_flow


class TestParseFlow:
    def test_minimal_flow(self, tmp_path):
        flow_file = tmp_path / "test.yaml"
        flow_file.write_text("""
name: test-flow
steps:
  - id: step1
    type: text.generate
    model: org/model
    input: "Hello"
""")
        flow = parse_flow(flow_file)
        assert flow.name == "test-flow"
        assert len(flow.steps) == 1
        assert flow.steps[0].id == "step1"
        assert flow.steps[0].type == "text.generate"
        assert flow.steps[0].model == "org/model"

    def test_full_flow(self, tmp_path):
        flow_file = tmp_path / "test.yaml"
        flow_file.write_text("""
name: transcribe-summarize
description: Transcribe and summarize
schedule: "0 2 * * *"
config:
  retries: 3
  retry_delay: 60
  concurrency: 4
  cache: true
steps:
  - id: transcribe
    type: audio.transcribe
    model: whisper
    input: $input
  - id: summarize
    type: text.generate
    model: llama
    input: "Summarize: {{ steps.transcribe.output }}"
output: $steps.summarize.output
""")
        flow = parse_flow(flow_file)
        assert flow.name == "transcribe-summarize"
        assert flow.description == "Transcribe and summarize"
        assert flow.schedule == "0 2 * * *"
        assert flow.config.retries == 3
        assert flow.config.cache is True
        assert len(flow.steps) == 2
        assert flow.output == "$steps.summarize.output"

    def test_no_name(self, tmp_path):
        flow_file = tmp_path / "test.yaml"
        flow_file.write_text("""
steps:
  - id: step1
    type: text.generate
""")
        with pytest.raises(ValueError, match="must have a 'name'"):
            parse_flow(flow_file)

    def test_no_steps(self, tmp_path):
        flow_file = tmp_path / "test.yaml"
        flow_file.write_text("""
name: empty
steps: []
""")
        with pytest.raises(ValueError, match="at least one step"):
            parse_flow(flow_file)

    def test_step_no_id(self, tmp_path):
        flow_file = tmp_path / "test.yaml"
        flow_file.write_text("""
name: test
steps:
  - type: text.generate
""")
        with pytest.raises(ValueError, match="must have an 'id'"):
            parse_flow(flow_file)

    def test_invalid_yaml(self, tmp_path):
        flow_file = tmp_path / "test.yaml"
        flow_file.write_text("just a string")
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            parse_flow(flow_file)

    def test_step_with_node(self, tmp_path):
        flow_file = tmp_path / "test.yaml"
        flow_file.write_text("""
name: remote
steps:
  - id: generate
    type: text.generate
    model: llama
    node: gpu-server
    input: $input
""")
        flow = parse_flow(flow_file)
        assert flow.steps[0].node == "gpu-server"

    def test_step_without_node(self, tmp_path):
        flow_file = tmp_path / "test.yaml"
        flow_file.write_text("""
name: local
steps:
  - id: generate
    type: text.generate
    model: llama
    input: $input
""")
        flow = parse_flow(flow_file)
        assert flow.steps[0].node is None

    def test_map_step(self, tmp_path):
        flow_file = tmp_path / "test.yaml"
        flow_file.write_text("""
name: batch
steps:
  - id: process
    type: text.generate
    model: llama
    input: $input
    map: true
""")
        flow = parse_flow(flow_file)
        assert flow.steps[0].map is True


class TestListFlows:
    def test_empty_dir(self, tmp_path):
        assert list_flows(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert list_flows(tmp_path / "nope") == []

    def test_none_dir(self):
        assert list_flows(None) == []

    def test_multiple_flows(self, tmp_path):
        (tmp_path / "a.yaml").write_text("""
name: flow-a
steps:
  - id: s1
    type: text.generate
""")
        (tmp_path / "b.yaml").write_text("""
name: flow-b
steps:
  - id: s1
    type: audio.transcribe
""")
        flows = list_flows(tmp_path)
        assert len(flows) == 2
        names = {f.name for f in flows}
        assert names == {"flow-a", "flow-b"}

    def test_skips_invalid(self, tmp_path):
        (tmp_path / "good.yaml").write_text("""
name: good
steps:
  - id: s1
    type: text.generate
""")
        (tmp_path / "bad.yaml").write_text("not valid")
        flows = list_flows(tmp_path)
        assert len(flows) == 1
        assert flows[0].name == "good"
