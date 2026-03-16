"""Tests for flow execution engine."""

from __future__ import annotations

import pytest

from silo.flows.parser import FlowConfig, FlowDefinition, FlowStep
from silo.flows.runner import FlowResult, StepResult, _resolve_input, run_flow


class TestResolveInput:
    def test_none_returns_input_data(self):
        assert _resolve_input(None, "hello", {}) == "hello"

    def test_dollar_input(self):
        assert _resolve_input("$input", "hello", {}) == "hello"

    def test_steps_reference(self):
        results = {"transcribe": "some text"}
        assert _resolve_input("$steps.transcribe.output", None, results) == "some text"

    def test_literal_string(self):
        assert _resolve_input("plain text", None, {}) == "plain text"


class TestRunFlow:
    def test_glob_step(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        flow = FlowDefinition(
            name="test",
            steps=[FlowStep(id="glob", type="fs.glob", input="*.txt")],
        )
        result = run_flow(flow, input_data="*.txt")
        assert result.success
        assert len(result.step_results) == 1
        assert result.step_results[0].success

    def test_write_step(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        flow = FlowDefinition(
            name="test",
            steps=[FlowStep(id="write", type="fs.write")],
        )
        result = run_flow(flow, input_data={"content": "hello", "path": str(tmp_path / "out.txt")})
        assert result.success
        assert (tmp_path / "out.txt").read_text() == "hello"

    def test_unknown_step_type(self):
        flow = FlowDefinition(
            name="test",
            steps=[FlowStep(id="bad", type="unknown.type")],
        )
        result = run_flow(flow)
        assert not result.success
        assert "Unknown step type" in (result.error or "")

    def test_stt_step_not_implemented(self):
        flow = FlowDefinition(
            name="test",
            steps=[FlowStep(id="stt", type="audio.transcribe", model="whisper")],
        )
        result = run_flow(flow, input_data="audio.wav")
        assert not result.success
        assert "not yet connected" in (result.error or "")

    def test_chat_step_not_implemented(self):
        flow = FlowDefinition(
            name="test",
            steps=[FlowStep(id="chat", type="text.generate", model="llama")],
        )
        result = run_flow(flow, input_data="Hello")
        assert not result.success
        assert "not yet connected" in (result.error or "")

    def test_multi_step_flow(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.txt").write_text("content")

        flow = FlowDefinition(
            name="test",
            steps=[
                FlowStep(id="glob", type="fs.glob", input="*.txt"),
            ],
            output="$steps.glob.output",
        )
        result = run_flow(flow)
        assert result.success
        assert result.final_output is not None


class TestFlowResult:
    def test_result_dataclass(self):
        r = FlowResult(flow_name="test", success=True)
        assert r.flow_name == "test"
        assert r.success is True
        assert r.step_results == []

    def test_step_result(self):
        s = StepResult(step_id="s1", success=True, output="done")
        assert s.step_id == "s1"
        assert s.output == "done"
