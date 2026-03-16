"""Backend protocol definitions — one per modality."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseBackend(Protocol):
    """Shared lifecycle methods for all backends."""

    def load(self, model_path: str, config: dict) -> None:  # type: ignore[type-arg]
        """Load a model into memory."""
        ...

    def unload(self) -> None:
        """Unload model from memory."""
        ...

    def health(self) -> dict:  # type: ignore[type-arg]
        """Return health status dict."""
        ...


@runtime_checkable
class ChatBackend(BaseBackend, Protocol):
    """Backend for LLM chat completions."""

    def chat(
        self,
        messages: list[dict],  # type: ignore[type-arg]
        stream: bool = False,
        **kwargs: object,
    ) -> Iterator[dict] | dict:  # type: ignore[type-arg]
        """OpenAI-compatible chat completion."""
        ...


@runtime_checkable
class SttBackend(BaseBackend, Protocol):
    """Backend for speech-to-text."""

    def transcribe(
        self,
        audio: bytes,
        language: str | None = None,
        response_format: str = "json",
        content_type: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        """OpenAI-compatible audio transcription."""
        ...


@runtime_checkable
class TtsBackend(BaseBackend, Protocol):
    """Backend for text-to-speech."""

    def speak(
        self,
        text: str,
        voice: str = "default",
        response_format: str = "wav",
        speed: float = 1.0,
        stream: bool = False,
    ) -> bytes | Iterator[bytes]:
        """OpenAI-compatible text-to-speech."""
        ...

    def voices(self) -> list[dict[str, str]]:
        """Return available voices as [{"id": "...", "name": "..."}]."""
        ...
