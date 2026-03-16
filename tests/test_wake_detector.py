"""Tests for wake word detector."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from silo.wake.detector import DetectorConfig, WakeWordDetector


def _oww_modules(mock_oww):
    """Build the sys.modules dict for patching openwakeword."""
    return {
        "openwakeword": mock_oww,
        "openwakeword.model": mock_oww.model,
    }


class TestDetectorConfig:
    def test_defaults(self):
        config = DetectorConfig()
        assert config.wake_word == "hey_jarvis"
        assert config.threshold == 0.5
        assert config.model_path is None

    def test_frozen(self):
        config = DetectorConfig()
        with pytest.raises(AttributeError):
            config.wake_word = "alexa"  # type: ignore[misc]

    def test_custom_values(self):
        config = DetectorConfig(
            wake_word="alexa",
            threshold=0.8,
            model_path="/path/to/model.onnx",
        )
        assert config.wake_word == "alexa"
        assert config.threshold == 0.8
        assert config.model_path == "/path/to/model.onnx"


class TestWakeWordDetector:
    def test_feed_without_load_raises(self):
        detector = WakeWordDetector()
        with pytest.raises(RuntimeError, match="not loaded"):
            detector.feed(MagicMock())

    def test_load_downloads_models(self):
        mock_oww = MagicMock()
        with patch.dict(sys.modules, _oww_modules(mock_oww)):
            detector = WakeWordDetector()
            detector.load()
            mock_oww.utils.download_models.assert_called_once()

    def test_feed_returns_false_below_threshold(self):
        mock_oww = MagicMock()
        mock_oww.model.Model.return_value.predict.return_value = {
            "hey_jarvis": 0.2,
        }
        with patch.dict(sys.modules, _oww_modules(mock_oww)):
            detector = WakeWordDetector(DetectorConfig(threshold=0.5))
            detector.load()
            assert detector.feed(MagicMock()) is False

    def test_feed_returns_true_above_threshold(self):
        mock_oww = MagicMock()
        mock_oww.model.Model.return_value.predict.return_value = {
            "hey_jarvis": 0.8,
        }
        with patch.dict(sys.modules, _oww_modules(mock_oww)):
            detector = WakeWordDetector(DetectorConfig(threshold=0.5))
            detector.load()
            assert detector.feed(MagicMock()) is True

    def test_feed_matches_partial_model_name(self):
        mock_oww = MagicMock()
        mock_oww.model.Model.return_value.predict.return_value = {
            "models/hey_jarvis_v0.1": 0.9,
        }
        with patch.dict(sys.modules, _oww_modules(mock_oww)):
            detector = WakeWordDetector(
                DetectorConfig(wake_word="hey_jarvis")
            )
            detector.load()
            assert detector.feed(MagicMock()) is True

    def test_feed_no_match_wrong_word(self):
        mock_oww = MagicMock()
        mock_oww.model.Model.return_value.predict.return_value = {
            "alexa": 0.9,
        }
        with patch.dict(sys.modules, _oww_modules(mock_oww)):
            detector = WakeWordDetector(
                DetectorConfig(wake_word="hey_jarvis")
            )
            detector.load()
            assert detector.feed(MagicMock()) is False

    def test_reset_calls_model_reset(self):
        mock_oww = MagicMock()
        mock_model = mock_oww.model.Model.return_value
        with patch.dict(sys.modules, _oww_modules(mock_oww)):
            detector = WakeWordDetector()
            detector.load()
            detector.reset()
            mock_model.reset.assert_called_once()

    def test_custom_model_path(self):
        mock_oww = MagicMock()
        with patch.dict(sys.modules, _oww_modules(mock_oww)):
            config = DetectorConfig(model_path="/path/to/custom.onnx")
            detector = WakeWordDetector(config)
            detector.load()
            call_kwargs = mock_oww.model.Model.call_args[1]
            assert call_kwargs["wakeword_models"] == [
                "/path/to/custom.onnx"
            ]
