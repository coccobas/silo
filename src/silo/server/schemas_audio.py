"""OpenAI-compatible audio request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptionRequest(BaseModel):
    """OpenAI-compatible audio transcription request (form fields)."""

    model: str
    language: str | None = None
    response_format: str = "json"


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""

    text: str


class TranscriptionVerboseResponse(BaseModel):
    """OpenAI-compatible verbose transcription response."""

    text: str
    language: str = ""
    duration: float = 0.0
    segments: list[dict] = Field(default_factory=list)  # type: ignore[type-arg]


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request."""

    model: str
    input: str
    voice: str = "default"
    response_format: str = "mp3"
    speed: float = 1.0


class VoiceEntry(BaseModel):
    """A single voice entry."""

    id: str
    name: str


class VoicesResponse(BaseModel):
    """Response for GET /v1/audio/voices."""

    voices: list[VoiceEntry]


class AudioModelEntry(BaseModel):
    """A single audio model entry."""

    id: str


class AudioModelsResponse(BaseModel):
    """Response for GET /v1/audio/models."""

    models: list[AudioModelEntry]
