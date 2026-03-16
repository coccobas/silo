"""Tests for chat completion routes."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from silo.server.app import create_app


class MockChatBackend:
    """Mock backend for testing."""

    def __init__(self, response_text: str = "Hello from mock!") -> None:
        self._response_text = response_text

    def load(self, model_path: str, config: dict) -> None:
        pass

    def unload(self) -> None:
        pass

    def health(self) -> dict:
        return {"status": "ok", "model": "test-model", "backend": "mock"}

    def chat(self, messages: list[dict], stream: bool = False, **kwargs) -> dict:
        if stream:
            return self._stream(messages, **kwargs)
        return {
            "choices": [{"message": {"role": "assistant", "content": self._response_text}}],
            "model": "test-model",
        }

    def _stream(self, messages: list[dict], **kwargs):
        words = self._response_text.split()
        for word in words:
            yield {"choices": [{"delta": {"content": word + " "}}]}


@pytest.fixture
def app():
    backend = MockChatBackend()
    return create_app(backend, "test-model")


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
class TestChatCompletions:
    async def test_non_streaming(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test-model"
        assert data["choices"][0]["message"]["content"] == "Hello from mock!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data
        assert data["usage"]["total_tokens"] > 0
        assert data["id"].startswith("chatcmpl-")

    async def test_streaming(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200

        # Parse SSE events
        lines = response.text.strip().split("\n")
        data_lines = [l.removeprefix("data: ") for l in lines if l.startswith("data: ")]

        # Should have content chunks + [DONE]
        assert len(data_lines) >= 2
        assert data_lines[-1] == "[DONE]"

        # Parse first content chunk
        chunk = json.loads(data_lines[0])
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "test-model"
        assert "delta" in chunk["choices"][0]

    async def test_model_mismatch(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "wrong-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert response.status_code == 404
        data = response.json()
        assert data["error"]["type"] == "model_not_found"

    async def test_empty_messages(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [],
            },
        )
        # Empty messages is valid per schema, backend handles it
        assert response.status_code == 200

    async def test_missing_model_field(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert response.status_code == 422  # Pydantic validation

    async def test_custom_params(self, client):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.5,
                "max_tokens": 100,
            },
        )
        assert response.status_code == 200


class MockErrorBackend(MockChatBackend):
    def chat(self, messages, stream=False, **kwargs):
        raise RuntimeError("Backend crashed")


@pytest.mark.asyncio
class TestChatErrors:
    async def test_backend_error(self):
        app = create_app(MockErrorBackend(), "test-model")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert response.status_code == 500
            data = response.json()
            assert data["error"]["type"] == "backend_error"
            assert "crashed" in data["error"]["message"].lower()
