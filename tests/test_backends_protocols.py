"""Tests for backend protocols."""

from collections.abc import Iterator

from silo.backends.protocols import BaseBackend, ChatBackend, SttBackend, TtsBackend


class MockChatBackend:
    def load(self, model_path: str, config: dict) -> None:
        pass

    def unload(self) -> None:
        pass

    def health(self) -> dict:
        return {"status": "ok"}

    def chat(self, messages: list[dict], stream: bool = False, **kwargs) -> dict:
        return {"choices": [{"message": {"content": "hello"}}]}


class MockSttBackend:
    def load(self, model_path: str, config: dict) -> None:
        pass

    def unload(self) -> None:
        pass

    def health(self) -> dict:
        return {"status": "ok"}

    def transcribe(self, audio: bytes, language=None, response_format="json") -> dict:
        return {"text": "transcribed"}


class MockTtsBackend:
    def load(self, model_path: str, config: dict) -> None:
        pass

    def unload(self) -> None:
        pass

    def health(self) -> dict:
        return {"status": "ok"}

    def speak(self, text: str, voice="default", response_format="wav", speed=1.0, stream=False) -> bytes:
        return b"audio"

    def voices(self) -> list[dict[str, str]]:
        return [{"id": "default", "name": "Default"}]


class TestProtocols:
    def test_chat_backend_compliance(self):
        backend = MockChatBackend()
        assert isinstance(backend, BaseBackend)
        assert isinstance(backend, ChatBackend)

    def test_stt_backend_compliance(self):
        backend = MockSttBackend()
        assert isinstance(backend, BaseBackend)
        assert isinstance(backend, SttBackend)

    def test_tts_backend_compliance(self):
        backend = MockTtsBackend()
        assert isinstance(backend, BaseBackend)
        assert isinstance(backend, TtsBackend)

    def test_chat_not_stt(self):
        backend = MockChatBackend()
        assert not isinstance(backend, SttBackend)

    def test_stt_not_tts(self):
        backend = MockSttBackend()
        assert not isinstance(backend, TtsBackend)
