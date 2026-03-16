"""Tests for wake word listener orchestrator."""

from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest

from silo.wake.listener import ListenerConfig, WakeState, WakeStatus, WakeWordListener


class TestListenerConfig:
    def test_defaults(self):
        config = ListenerConfig()
        assert config.wake_word == "hey_jarvis"
        assert config.threshold == 0.5
        assert config.continuous is True
        assert config.sample_rate == 16000

    def test_frozen(self):
        config = ListenerConfig()
        with pytest.raises(AttributeError):
            config.wake_word = "test"  # type: ignore[misc]


class TestWakeStatus:
    def test_creation(self):
        status = WakeStatus(
            state=WakeState.LISTENING,
            wake_word="hey_jarvis",
            flow_name="test-flow",
        )
        assert status.state == WakeState.LISTENING
        assert status.detected_at is None
        assert status.detections == 0


class TestWakeWordListener:
    def _make_listener(
        self,
        flow_runner=None,
        on_status=None,
        continuous=True,
    ):
        config = ListenerConfig(
            wake_word="hey_jarvis",
            flow_name="test-flow",
            continuous=continuous,
        )
        return WakeWordListener(
            config=config,
            flow_runner=flow_runner or MagicMock(),
            on_status=on_status,
        )

    @patch("silo.wake.listener.WakeWordDetector")
    @patch("silo.wake.listener.AudioCapture")
    def test_detection_triggers_flow(self, mock_capture_cls, mock_detector_cls):
        """When the detector returns True, the flow runner should be called."""
        flow_runner = MagicMock()
        statuses: list[WakeStatus] = []

        mock_capture = MagicMock()
        mock_capture_cls.return_value = mock_capture

        # Simulate: one chunk that triggers detection, then stop
        audio_queue = queue.Queue()
        audio_queue.put(MagicMock())  # One chunk
        mock_capture.queue = audio_queue

        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_detector.feed.return_value = True

        listener = self._make_listener(
            flow_runner=flow_runner,
            on_status=statuses.append,
            continuous=False,  # Stop after first detection
        )
        # Replace internals with mocks
        listener._capture = mock_capture
        listener._detector = mock_detector

        listener.run()

        flow_runner.assert_called_once_with("test-flow")
        assert listener.detections == 1

        # Verify state transitions
        states = [s.state for s in statuses]
        assert WakeState.LISTENING in states
        assert WakeState.DETECTED in states
        assert WakeState.RUNNING_FLOW in states
        assert WakeState.STOPPED in states

    @patch("silo.wake.listener.WakeWordDetector")
    @patch("silo.wake.listener.AudioCapture")
    def test_no_detection_when_below_threshold(self, mock_capture_cls, mock_detector_cls):
        """When the detector returns False, no flow should be triggered."""
        flow_runner = MagicMock()

        mock_capture = MagicMock()
        mock_capture_cls.return_value = mock_capture

        audio_queue = queue.Queue()
        audio_queue.put(MagicMock())
        mock_capture.queue = audio_queue

        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_detector.feed.return_value = False

        listener = self._make_listener(flow_runner=flow_runner, continuous=False)
        listener._capture = mock_capture
        listener._detector = mock_detector

        # Stop the listener after a brief moment
        def stop_later():
            import time
            time.sleep(0.3)
            listener.stop()

        stopper = threading.Thread(target=stop_later)
        stopper.start()
        listener.run()
        stopper.join()

        flow_runner.assert_not_called()
        assert listener.detections == 0

    @patch("silo.wake.listener.WakeWordDetector")
    @patch("silo.wake.listener.AudioCapture")
    def test_continuous_mode_rearming(self, mock_capture_cls, mock_detector_cls):
        """In continuous mode, listener re-arms after flow completes."""
        call_count = 0
        flow_runner = MagicMock()

        mock_capture = MagicMock()
        mock_capture_cls.return_value = mock_capture

        # Two chunks that both trigger detection
        audio_queue = queue.Queue()
        audio_queue.put(MagicMock())
        audio_queue.put(MagicMock())
        mock_capture.queue = audio_queue

        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector

        def feed_side_effect(chunk):
            nonlocal call_count
            call_count += 1
            return True

        mock_detector.feed.side_effect = feed_side_effect

        listener = self._make_listener(flow_runner=flow_runner, continuous=True)
        listener._capture = mock_capture
        listener._detector = mock_detector

        # Stop after second detection
        original_flow = flow_runner

        def flow_with_stop(name):
            original_flow(name)
            if original_flow.call_count >= 2:
                listener.stop()

        listener._flow_runner = flow_with_stop

        listener.run()

        assert original_flow.call_count == 2
        assert listener.detections == 2
        # Detector should be reset between detections
        assert mock_detector.reset.call_count >= 2

    @patch("silo.wake.listener.WakeWordDetector")
    @patch("silo.wake.listener.AudioCapture")
    def test_stop_event(self, mock_capture_cls, mock_detector_cls):
        """Calling stop() should terminate the listener."""
        mock_capture = MagicMock()
        mock_capture_cls.return_value = mock_capture
        mock_capture.queue = queue.Queue()

        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector

        listener = self._make_listener()
        listener._capture = mock_capture
        listener._detector = mock_detector

        def stop_later():
            import time
            time.sleep(0.2)
            listener.stop()

        stopper = threading.Thread(target=stop_later)
        stopper.start()
        listener.run()
        stopper.join()

    @patch("silo.wake.listener.WakeWordDetector")
    @patch("silo.wake.listener.AudioCapture")
    def test_flow_error_emits_error_state(self, mock_capture_cls, mock_detector_cls):
        """If the flow runner raises, error state should be emitted."""
        statuses: list[WakeStatus] = []

        mock_capture = MagicMock()
        mock_capture_cls.return_value = mock_capture
        audio_queue = queue.Queue()
        audio_queue.put(MagicMock())
        mock_capture.queue = audio_queue

        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_detector.feed.return_value = True

        flow_runner = MagicMock(side_effect=RuntimeError("flow broke"))

        listener = self._make_listener(
            flow_runner=flow_runner,
            on_status=statuses.append,
            continuous=False,
        )
        listener._capture = mock_capture
        listener._detector = mock_detector

        listener.run()

        error_states = [s for s in statuses if s.state == WakeState.ERROR]
        assert len(error_states) >= 1
        assert "flow broke" in (error_states[0].error or "")
