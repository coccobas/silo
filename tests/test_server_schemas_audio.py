"""Tests for audio request/response schemas."""

from silo.server.schemas_audio import (
    SpeechRequest,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionVerboseResponse,
)


class TestTranscriptionRequest:
    def test_minimal(self):
        req = TranscriptionRequest(model="whisper")
        assert req.model == "whisper"
        assert req.language is None
        assert req.response_format == "json"

    def test_full(self):
        req = TranscriptionRequest(
            model="whisper", language="en", response_format="verbose_json"
        )
        assert req.language == "en"
        assert req.response_format == "verbose_json"


class TestTranscriptionResponse:
    def test_response(self):
        resp = TranscriptionResponse(text="Hello world")
        assert resp.text == "Hello world"


class TestTranscriptionVerboseResponse:
    def test_defaults(self):
        resp = TranscriptionVerboseResponse(text="Hello")
        assert resp.text == "Hello"
        assert resp.language == ""
        assert resp.duration == 0.0
        assert resp.segments == []

    def test_full(self):
        resp = TranscriptionVerboseResponse(
            text="Hello",
            language="en",
            duration=2.5,
            segments=[{"start": 0.0, "end": 2.5, "text": "Hello"}],
        )
        assert len(resp.segments) == 1


class TestSpeechRequest:
    def test_minimal(self):
        req = SpeechRequest(model="tts", input="Hello")
        assert req.model == "tts"
        assert req.input == "Hello"
        assert req.voice == "default"
        assert req.response_format == "mp3"
        assert req.speed == 1.0

    def test_full(self):
        req = SpeechRequest(
            model="tts",
            input="Hello",
            voice="alloy",
            response_format="mp3",
            speed=1.5,
        )
        assert req.voice == "alloy"
        assert req.response_format == "mp3"
        assert req.speed == 1.5
