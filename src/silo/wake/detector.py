"""Wake word detector — wraps openwakeword for keyword detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for wake word detection."""

    wake_word: str = "hey_jarvis"
    threshold: float = 0.5
    model_path: str | None = None  # None = use built-in model


class WakeWordDetector:
    """Detect a wake word in audio chunks using openwakeword."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()
        self._model: Any = None

    @property
    def config(self) -> DetectorConfig:
        return self._config

    def load(self) -> None:
        """Load the openwakeword model."""
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as e:
            raise ImportError(
                "Wake word detection requires openwakeword. "
                "Install with: silo setup install wake"
            ) from e

        # Download default models if needed
        openwakeword.utils.download_models()

        model_kwargs: dict[str, Any] = {
            "inference_framework": "onnx",
        }
        if self._config.model_path:
            model_kwargs["wakeword_models"] = [self._config.model_path]

        self._model = Model(**model_kwargs)

    def feed(self, chunk: Any) -> bool:
        """Feed an audio chunk and return True if wake word detected.

        Args:
            chunk: numpy int16 array of audio samples.

        Returns:
            True if the wake word score exceeds the threshold.
        """
        if self._model is None:
            raise RuntimeError("Detector not loaded. Call load() first.")

        prediction = self._model.predict(chunk)

        # Check if any model name matches our wake word
        for model_name, score in prediction.items():
            if self._config.wake_word in model_name and score >= self._config.threshold:
                return True

        return False

    def reset(self) -> None:
        """Reset internal state to prevent double-triggers."""
        if self._model is not None:
            self._model.reset()
