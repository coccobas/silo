"""Registry data models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ModelFormat(StrEnum):
    MLX = "mlx"
    GGUF = "gguf"
    STANDARD = "standard"
    AUDIO_STT = "audio-stt"
    AUDIO_TTS = "audio-tts"
    UNKNOWN = "unknown"


class RegistryEntry(BaseModel):
    """A model tracked in the local registry."""

    model_config = {"frozen": True}

    repo_id: str
    format: ModelFormat = ModelFormat.UNKNOWN
    local_path: str | None = None
    size_bytes: int | None = None
    downloaded_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    tags: list[str] = Field(default_factory=list)
