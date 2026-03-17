"""Tests for the flow creation modal and save_flow."""

from __future__ import annotations

import pytest

from silo.flows.parser import FlowDefinition, FlowStep, parse_flow, save_flow
from silo.tui.widgets.flow_create_modal import FlowCreateModal, FlowDraft, FlowStepDraft


class TestFlowStepDraft:
    def test_creation(self):
        step = FlowStepDraft(
            id="transcribe",
            type="audio.transcribe",
            model="mlx-community/whisper-large-v3-turbo",
            input="$input",
        )
        assert step.id == "transcribe"
        assert step.type == "audio.transcribe"
        assert step.model == "mlx-community/whisper-large-v3-turbo"
        assert step.input == "$input"

    def test_immutable(self):
        step = FlowStepDraft(
            id="test", type="fs.glob", model="", input=""
        )
        with pytest.raises(AttributeError):
            step.id = "changed"


class TestFlowDraft:
    def test_creation(self):
        draft = FlowDraft(
            name="my-flow",
            description="A test flow",
            steps=[
                FlowStepDraft(
                    id="step1",
                    type="text.generate",
                    model="some-model",
                    input="hello",
                ),
            ],
        )
        assert draft.name == "my-flow"
        assert draft.description == "A test flow"
        assert len(draft.steps) == 1

    def test_immutable(self):
        draft = FlowDraft(name="test", description="", steps=[])
        with pytest.raises(AttributeError):
            draft.name = "changed"

    def test_default_steps(self):
        draft = FlowDraft(name="test", description="")
        assert draft.steps == []


class TestSaveFlow:
    def test_save_and_reload(self, tmp_path):
        flow = FlowDefinition(
            name="test-flow",
            description="A test",
            steps=[
                FlowStep(
                    id="greet",
                    type="text.generate",
                    model="test-model",
                    input="Say hello",
                ),
            ],
            output="$steps.greet.output",
        )

        path = save_flow(flow, tmp_path)
        assert path.exists()
        assert path.name == "test-flow.yaml"

        reloaded = parse_flow(path)
        assert reloaded.name == "test-flow"
        assert reloaded.description == "A test"
        assert len(reloaded.steps) == 1
        assert reloaded.steps[0].id == "greet"
        assert reloaded.steps[0].type == "text.generate"
        assert reloaded.steps[0].model == "test-model"
        assert reloaded.steps[0].input == "Say hello"
        assert reloaded.output == "$steps.greet.output"

    def test_save_creates_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "flows"
        flow = FlowDefinition(
            name="nested",
            steps=[FlowStep(id="s1", type="fs.glob")],
        )
        path = save_flow(flow, nested)
        assert path.exists()
        assert nested.exists()

    def test_save_minimal_flow(self, tmp_path):
        flow = FlowDefinition(
            name="minimal",
            steps=[FlowStep(id="s1", type="fs.glob")],
        )
        path = save_flow(flow, tmp_path)

        reloaded = parse_flow(path)
        assert reloaded.name == "minimal"
        assert reloaded.description == ""
        assert len(reloaded.steps) == 1
        assert reloaded.steps[0].model is None
        assert reloaded.output is None

    def test_save_multi_step_flow(self, tmp_path):
        flow = FlowDefinition(
            name="multi",
            description="Multi-step",
            steps=[
                FlowStep(
                    id="transcribe",
                    type="audio.transcribe",
                    model="whisper",
                    input="$input",
                ),
                FlowStep(
                    id="summarize",
                    type="text.generate",
                    model="llama",
                    input="$steps.transcribe.output",
                ),
            ],
            output="$steps.summarize.output",
        )
        path = save_flow(flow, tmp_path)

        reloaded = parse_flow(path)
        assert len(reloaded.steps) == 2
        assert reloaded.steps[0].id == "transcribe"
        assert reloaded.steps[1].id == "summarize"
        assert reloaded.steps[1].input == "$steps.transcribe.output"

    def test_save_overwrites_existing(self, tmp_path):
        flow_v1 = FlowDefinition(
            name="evolve",
            steps=[FlowStep(id="s1", type="fs.glob")],
        )
        save_flow(flow_v1, tmp_path)

        flow_v2 = FlowDefinition(
            name="evolve",
            description="Updated",
            steps=[
                FlowStep(id="s1", type="fs.glob"),
                FlowStep(id="s2", type="fs.write"),
            ],
        )
        save_flow(flow_v2, tmp_path)

        reloaded = parse_flow(tmp_path / "evolve.yaml")
        assert reloaded.description == "Updated"
        assert len(reloaded.steps) == 2


class TestFlowCreateModal:
    def test_init(self):
        modal = FlowCreateModal()
        assert modal._steps == []
