"""Tests for OpenAI-compatible schemas."""

import pytest
from pydantic import ValidationError

from silo.server.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaContent,
    HealthResponse,
    ModelListResponse,
    ModelObject,
    StreamChoice,
    Usage,
)


class TestChatCompletionRequest:
    def test_minimal(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
        )
        assert req.model == "test"
        assert req.temperature == 0.7
        assert req.stream is False
        assert req.max_tokens == 512

    def test_full(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            temperature=0.5,
            max_tokens=100,
            stream=True,
            stop=["\\n"],
        )
        assert req.stream is True
        assert req.stop == ["\\n"]

    def test_missing_model(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])  # type: ignore[call-arg]

    def test_missing_messages(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="test")  # type: ignore[call-arg]


class TestChatCompletionResponse:
    def test_response(self):
        resp = ChatCompletionResponse(
            model="test",
            choices=[Choice(message=ChoiceMessage(content="hello"))],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        assert resp.object == "chat.completion"
        assert resp.model == "test"
        assert resp.choices[0].message.content == "hello"
        assert resp.id.startswith("chatcmpl-")
        assert resp.created > 0


class TestChatCompletionChunk:
    def test_chunk(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            model="test",
            choices=[StreamChoice(delta=DeltaContent(content="hi"))],
        )
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.content == "hi"

    def test_final_chunk(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            model="test",
            choices=[StreamChoice(delta=DeltaContent(), finish_reason="stop")],
        )
        assert chunk.choices[0].finish_reason == "stop"


class TestModelSchemas:
    def test_model_object(self):
        model = ModelObject(id="test-model")
        assert model.object == "model"
        assert model.owned_by == "silo"

    def test_model_list(self):
        resp = ModelListResponse(data=[ModelObject(id="m1"), ModelObject(id="m2")])
        assert resp.object == "list"
        assert len(resp.data) == 2

    def test_health_response(self):
        resp = HealthResponse(status="ok", model="test", backend="mlx-lm")
        assert resp.status == "ok"
