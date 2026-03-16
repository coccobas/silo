"""Audio routes — STT and TTS endpoints."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from silo.server.errors import (
    BACKEND_ERROR,
    INVALID_REQUEST,
    MODEL_NOT_FOUND,
    openai_error_response,
)
from silo.server.schemas_audio import (
    AudioModelEntry,
    AudioModelsResponse,
    SpeechRequest,
    TranscriptionResponse,
    TranscriptionVerboseResponse,
    VoiceEntry,
    VoicesResponse,
)

router = APIRouter()


@router.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    response_format: str = Form("json"),
):  # type: ignore[no-untyped-def]
    """OpenAI-compatible audio transcription endpoint."""
    backend = request.app.state.backend
    model_name = request.app.state.model_name

    if model != model_name:
        return openai_error_response(
            404,
            f"Model '{model}' not found. This server serves '{model_name}'.",
            MODEL_NOT_FOUND,
        )

    if not hasattr(backend, "transcribe"):
        return openai_error_response(
            400,
            "This model does not support audio transcription.",
            INVALID_REQUEST,
        )

    try:
        audio_bytes = await file.read()
        result = backend.transcribe(
            audio=audio_bytes,
            language=language,
            response_format=response_format,
            content_type=file.content_type,
        )
    except Exception as e:
        return openai_error_response(500, str(e), BACKEND_ERROR)

    if response_format == "text":
        return Response(content=result.get("text", ""), media_type="text/plain")

    if response_format == "verbose_json":
        return TranscriptionVerboseResponse(
            text=result.get("text", ""),
            language=result.get("language", ""),
            duration=result.get("duration", 0.0),
            segments=result.get("segments", []),
        )

    return TranscriptionResponse(text=result.get("text", ""))


@router.post("/v1/audio/speech")
async def speech(
    request_body: SpeechRequest,
    request: Request,
):  # type: ignore[no-untyped-def]
    """OpenAI-compatible text-to-speech endpoint."""
    backend = request.app.state.backend
    model_name = request.app.state.model_name

    if request_body.model != model_name:
        return openai_error_response(
            404,
            f"Model '{request_body.model}' not found. This server serves '{model_name}'.",
            MODEL_NOT_FOUND,
        )

    if not hasattr(backend, "speak"):
        return openai_error_response(
            400,
            "This model does not support text-to-speech.",
            INVALID_REQUEST,
        )

    content_type_map = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "pcm": "audio/pcm",
    }
    content_type = content_type_map.get(request_body.response_format, "audio/wav")

    try:
        result = backend.speak(
            text=request_body.input,
            voice=request_body.voice,
            response_format=request_body.response_format,
            speed=request_body.speed,
            stream=False,
        )
    except Exception as e:
        return openai_error_response(500, str(e), BACKEND_ERROR)

    if isinstance(result, bytes):
        return Response(content=result, media_type=content_type)

    # Streaming response
    return StreamingResponse(result, media_type=content_type)


@router.get("/v1/audio/voices")
async def list_voices(request: Request) -> VoicesResponse:
    """List available TTS voices. Used by Open WebUI for custom backends."""
    backend = request.app.state.backend

    if hasattr(backend, "voices"):
        raw_voices = backend.voices()
        return VoicesResponse(
            voices=[VoiceEntry(id=v["id"], name=v["name"]) for v in raw_voices]
        )

    # Fallback: no voice discovery, return a single default voice
    return VoicesResponse(voices=[VoiceEntry(id="default", name="Default")])


@router.get("/v1/audio/models")
async def list_audio_models(request: Request) -> AudioModelsResponse:
    """List available audio models. Used by Open WebUI for custom backends."""
    model_name = request.app.state.model_name
    return AudioModelsResponse(models=[AudioModelEntry(id=model_name)])
