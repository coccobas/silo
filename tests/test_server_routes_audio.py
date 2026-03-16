"""Tests for audio routes — STT and TTS endpoints."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from silo.server.app import create_app


class MockSttBackend:
    """Mock STT backend for testing."""

    def load(self, model_path: str, config: dict) -> None:
        pass

    def unload(self) -> None:
        pass

    def health(self) -> dict:
        return {"status": "ok", "model": "test-whisper", "backend": "mock-stt"}

    def transcribe(self, audio: bytes, language=None, response_format="json", content_type=None):
        result = {"text": "Hello world"}
        if response_format == "verbose_json":
            result.update({
                "language": language or "en",
                "duration": 2.5,
                "segments": [{"start": 0.0, "end": 2.5, "text": "Hello world"}],
            })
        return result


class MockTtsBackend:
    """Mock TTS backend for testing."""

    def load(self, model_path: str, config: dict) -> None:
        pass

    def unload(self) -> None:
        pass

    def health(self) -> dict:
        return {"status": "ok", "model": "test-tts", "backend": "mock-tts"}

    def speak(self, text, voice="default", response_format="wav", speed=1.0, stream=False):
        return b"fake audio bytes"

    def voices(self):
        return [
            {"id": "af_heart", "name": "Heart"},
            {"id": "am_adam", "name": "Adam"},
        ]


@pytest.fixture
def stt_app():
    return create_app(MockSttBackend(), "test-whisper")


@pytest.fixture
def tts_app():
    return create_app(MockTtsBackend(), "test-tts")


@pytest.fixture
async def stt_client(stt_app):
    transport = ASGITransport(app=stt_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def tts_client(tts_app):
    transport = ASGITransport(app=tts_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
class TestTranscriptionEndpoint:
    async def test_transcribe_json(self, stt_client):
        response = await stt_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio", "audio/wav")},
            data={"model": "test-whisper"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello world"

    async def test_transcribe_text_format(self, stt_client):
        response = await stt_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio", "audio/wav")},
            data={"model": "test-whisper", "response_format": "text"},
        )
        assert response.status_code == 200
        assert response.text == "Hello world"

    async def test_transcribe_verbose_json(self, stt_client):
        response = await stt_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio", "audio/wav")},
            data={"model": "test-whisper", "response_format": "verbose_json"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello world"
        assert data["language"] == "en"
        assert data["duration"] == 2.5
        assert len(data["segments"]) == 1

    async def test_transcribe_with_language(self, stt_client):
        response = await stt_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio", "audio/wav")},
            data={"model": "test-whisper", "language": "fr"},
        )
        assert response.status_code == 200

    async def test_transcribe_wrong_model(self, stt_client):
        response = await stt_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"fake audio", "audio/wav")},
            data={"model": "wrong-model"},
        )
        assert response.status_code == 404
        data = response.json()
        assert data["error"]["type"] == "model_not_found"


@pytest.mark.asyncio
class TestSpeechEndpoint:
    async def test_speech_basic(self, tts_client):
        response = await tts_client.post(
            "/v1/audio/speech",
            json={"model": "test-tts", "input": "Hello world"},
        )
        assert response.status_code == 200
        assert response.content == b"fake audio bytes"
        assert "audio" in response.headers["content-type"]

    async def test_speech_with_options(self, tts_client):
        response = await tts_client.post(
            "/v1/audio/speech",
            json={
                "model": "test-tts",
                "input": "Hello",
                "voice": "alloy",
                "response_format": "mp3",
                "speed": 1.5,
            },
        )
        assert response.status_code == 200

    async def test_speech_wrong_model(self, tts_client):
        response = await tts_client.post(
            "/v1/audio/speech",
            json={"model": "wrong-model", "input": "Hello"},
        )
        assert response.status_code == 404
        data = response.json()
        assert data["error"]["type"] == "model_not_found"


class MockChatOnlyBackend:
    """Backend that only supports chat, not audio."""

    def load(self, model_path, config):
        pass

    def unload(self):
        pass

    def health(self):
        return {"status": "ok", "model": "chat-only", "backend": "mock"}

    def chat(self, messages, stream=False, **kwargs):
        return {"choices": [{"message": {"content": "hi"}}]}


@pytest.mark.asyncio
class TestAudioOnChatBackend:
    async def test_transcribe_not_supported(self):
        """Chat-only backend should not have audio routes."""
        app = create_app(MockChatOnlyBackend(), "chat-only")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", b"audio", "audio/wav")},
                data={"model": "chat-only"},
            )
            # Should be 404 (route not registered) or 405
            assert response.status_code in (404, 405)

    async def test_speech_not_supported(self):
        """Chat-only backend should not have TTS route."""
        app = create_app(MockChatOnlyBackend(), "chat-only")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/audio/speech",
                json={"model": "chat-only", "input": "Hello"},
            )
            assert response.status_code in (404, 405)


@pytest.mark.asyncio
class TestSttBackendError:
    async def test_transcribe_error(self):
        class ErrorSttBackend(MockSttBackend):
            def transcribe(self, audio, language=None, response_format="json", content_type=None):
                raise RuntimeError("STT crashed")

        app = create_app(ErrorSttBackend(), "test-whisper")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", b"audio", "audio/wav")},
                data={"model": "test-whisper"},
            )
            assert response.status_code == 500
            data = response.json()
            assert data["error"]["type"] == "backend_error"


@pytest.mark.asyncio
class TestVoicesEndpoint:
    async def test_list_voices(self, tts_client):
        response = await tts_client.get("/v1/audio/voices")
        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert len(data["voices"]) == 2
        assert data["voices"][0]["id"] == "af_heart"
        assert data["voices"][0]["name"] == "Heart"

    async def test_voices_fallback_no_method(self):
        """Backend without voices() returns a default voice."""

        class MinimalTtsBackend:
            def load(self, model_path, config):
                pass

            def unload(self):
                pass

            def health(self):
                return {"status": "ok"}

            def speak(self, text, voice="default", response_format="wav", speed=1.0, stream=False):
                return b"audio"

        app = create_app(MinimalTtsBackend(), "minimal-tts")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.get("/v1/audio/voices")
            assert response.status_code == 200
            data = response.json()
            assert len(data["voices"]) == 1
            assert data["voices"][0]["id"] == "default"


@pytest.mark.asyncio
class TestAudioModelsEndpoint:
    async def test_list_audio_models(self, tts_client):
        response = await tts_client.get("/v1/audio/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 1
        assert data["models"][0]["id"] == "test-tts"

    async def test_stt_audio_models(self, stt_client):
        response = await stt_client.get("/v1/audio/models")
        assert response.status_code == 200
        data = response.json()
        assert data["models"][0]["id"] == "test-whisper"
