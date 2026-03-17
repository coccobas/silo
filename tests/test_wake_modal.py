"""Tests for the wake modal widget and dashboard wake integration."""

from __future__ import annotations

import pytest

from silo.tui.widgets.wake_modal import WakeModal, WakeSettings


class TestWakeSettings:
    def test_creation(self):
        settings = WakeSettings(
            wake_word="hey_jarvis",
            flow_name="test-flow",
            threshold=0.5,
            continuous=True,
            device=None,
        )
        assert settings.wake_word == "hey_jarvis"
        assert settings.flow_name == "test-flow"
        assert settings.threshold == 0.5
        assert settings.continuous is True
        assert settings.device is None

    def test_immutable(self):
        settings = WakeSettings(
            wake_word="hey_jarvis",
            flow_name="test-flow",
            threshold=0.5,
            continuous=True,
            device=None,
        )
        with pytest.raises(AttributeError):
            settings.wake_word = "alexa"


class TestWakeModal:
    def test_init_with_flows(self):
        modal = WakeModal(flow_names=["flow-a", "flow-b"])
        assert modal._flow_names == ["flow-a", "flow-b"]

    def test_init_empty_flows(self):
        modal = WakeModal(flow_names=[])
        assert modal._flow_names == []
