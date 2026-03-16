"""Tests for wake word audio capture."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from silo.wake.capture import AudioCapture, CaptureConfig


class TestCaptureConfig:
    def test_defaults(self):
        config = CaptureConfig()
        assert config.sample_rate == 16000
        assert config.chunk_size == 1280
        assert config.device is None

    def test_frozen(self):
        config = CaptureConfig()
        with pytest.raises(AttributeError):
            config.sample_rate = 44100  # type: ignore[misc]

    def test_custom_values(self):
        config = CaptureConfig(sample_rate=44100, chunk_size=2048, device=2)
        assert config.sample_rate == 44100
        assert config.chunk_size == 2048
        assert config.device == 2


class TestAudioCapture:
    def test_initial_state(self):
        capture = AudioCapture()
        assert not capture.is_active
        assert capture.queue.empty()

    def test_custom_config(self):
        config = CaptureConfig(sample_rate=44100)
        capture = AudioCapture(config)
        assert capture.config.sample_rate == 44100

    def test_start_opens_stream(self):
        mock_sd = MagicMock()
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_sd.InputStream.return_value = mock_stream

        with patch.dict(sys.modules, {"sounddevice": mock_sd, "numpy": MagicMock()}):
            capture = AudioCapture()
            capture.start()

            mock_sd.InputStream.assert_called_once()
            mock_stream.start.assert_called_once()
            assert capture.is_active

    def test_stop_closes_stream(self):
        mock_sd = MagicMock()
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_sd.InputStream.return_value = mock_stream

        with patch.dict(sys.modules, {"sounddevice": mock_sd, "numpy": MagicMock()}):
            capture = AudioCapture()
            capture.start()
            capture.stop()

            mock_stream.stop.assert_called_once()
            mock_stream.close.assert_called_once()
            assert not capture.is_active

    def test_stop_when_not_started(self):
        capture = AudioCapture()
        capture.stop()  # Should not raise

    def test_microphone_permission_error(self):
        mock_sd = MagicMock()
        mock_sd.InputStream.side_effect = Exception("PortAudio error: permission denied")

        with patch.dict(sys.modules, {"sounddevice": mock_sd, "numpy": MagicMock()}):
            capture = AudioCapture()
            with pytest.raises(RuntimeError, match="Microphone access denied"):
                capture.start()

    def test_drain_empty_queue(self):
        capture = AudioCapture()
        capture.drain()  # Should not raise

    def test_drain_clears_queue(self):
        capture = AudioCapture()
        capture.queue.put("chunk1")
        capture.queue.put("chunk2")
        assert not capture.queue.empty()
        capture.drain()
        assert capture.queue.empty()
