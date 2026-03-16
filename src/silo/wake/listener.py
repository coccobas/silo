"""Wake word listener — orchestrates capture, detection, and flow execution."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from time import time
from typing import Any

from silo.wake.capture import AudioCapture, CaptureConfig
from silo.wake.detector import DetectorConfig, WakeWordDetector


class WakeState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    DETECTED = "detected"
    RUNNING_FLOW = "running_flow"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass(frozen=True)
class WakeStatus:
    """Snapshot of the listener's current state."""

    state: WakeState
    wake_word: str
    flow_name: str
    detected_at: float | None = None
    error: str | None = None
    detections: int = 0


@dataclass(frozen=True)
class ListenerConfig:
    """Configuration for the wake word listener."""

    wake_word: str = "hey_jarvis"
    flow_name: str = ""
    threshold: float = 0.5
    model_path: str | None = None
    continuous: bool = True
    sample_rate: int = 16000
    chunk_size: int = 1280
    device: int | None = None


class WakeWordListener:
    """Orchestrates audio capture, wake word detection, and flow triggering."""

    def __init__(
        self,
        config: ListenerConfig,
        flow_runner: Callable[[str], Any],
        on_status: Callable[[WakeStatus], None] | None = None,
    ) -> None:
        self._config = config
        self._flow_runner = flow_runner
        self._on_status = on_status or (lambda _: None)
        self._stop_event = threading.Event()
        self._detections = 0

        self._capture = AudioCapture(CaptureConfig(
            sample_rate=config.sample_rate,
            chunk_size=config.chunk_size,
            device=config.device,
        ))
        self._detector = WakeWordDetector(DetectorConfig(
            wake_word=config.wake_word,
            threshold=config.threshold,
            model_path=config.model_path,
        ))

    @property
    def config(self) -> ListenerConfig:
        return self._config

    @property
    def detections(self) -> int:
        return self._detections

    def _emit(self, state: WakeState, **kwargs: Any) -> None:
        status = WakeStatus(
            state=state,
            wake_word=self._config.wake_word,
            flow_name=self._config.flow_name,
            detections=self._detections,
            **kwargs,
        )
        self._on_status(status)

    def run(self) -> None:
        """Start listening. Blocks until stop() is called or KeyboardInterrupt."""
        try:
            self._detector.load()
        except Exception as e:
            self._emit(WakeState.ERROR, error=str(e))
            raise

        try:
            self._capture.start()
        except Exception as e:
            self._emit(WakeState.ERROR, error=str(e))
            raise

        self._emit(WakeState.LISTENING)

        try:
            self._listen_loop()
        finally:
            self._capture.stop()
            self._emit(WakeState.STOPPED)

    def _listen_loop(self) -> None:
        """Main loop: pull audio chunks, feed to detector, trigger flow."""
        while not self._stop_event.is_set():
            try:
                chunk = self._capture.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._detector.feed(chunk):
                self._detections += 1
                detected_at = time()
                self._emit(WakeState.DETECTED, detected_at=detected_at)

                # Pause capture during flow execution to avoid queue buildup
                self._capture.stop()
                self._capture.drain()

                self._emit(WakeState.RUNNING_FLOW, detected_at=detected_at)
                try:
                    self._flow_runner(self._config.flow_name)
                except Exception as e:
                    self._emit(WakeState.ERROR, error=str(e))

                self._detector.reset()

                if not self._config.continuous or self._stop_event.is_set():
                    break

                # Re-arm: restart capture
                self._capture.start()
                self._emit(WakeState.LISTENING)

    def stop(self) -> None:
        """Signal the listener to stop."""
        self._stop_event.set()
