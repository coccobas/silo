"""Audio capture — stream microphone input into a queue for processing."""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CaptureConfig:
    """Configuration for microphone capture."""

    sample_rate: int = 16000
    chunk_size: int = 1280  # 80ms at 16kHz (openwakeword expectation)
    device: int | None = None  # None = system default


class AudioCapture:
    """Callback-based microphone capture using sounddevice."""

    def __init__(self, config: CaptureConfig | None = None) -> None:
        self._config = config or CaptureConfig()
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=200)
        self._stream: Any = None

    @property
    def config(self) -> CaptureConfig:
        return self._config

    @property
    def queue(self) -> queue.Queue[Any]:
        return self._queue

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active

    def start(self) -> None:
        """Open the microphone stream."""
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "Wake word detection requires sounddevice and numpy. "
                "Install with: uv pip install 'silo[wake]'"
            ) from e

        def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if status:
                return
            self._queue.put(indata[:, 0].copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self._config.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self._config.chunk_size,
                device=self._config.device,
                callback=callback,
            )
            self._stream.start()
        except Exception as e:
            error_msg = str(e)
            if "PortAudio" in error_msg or "permission" in error_msg.lower():
                raise RuntimeError(
                    "Microphone access denied. Grant permission in "
                    "System Settings > Privacy & Security > Microphone."
                ) from e
            raise

    def stop(self) -> None:
        """Close the microphone stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def drain(self) -> None:
        """Discard all queued audio chunks."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
