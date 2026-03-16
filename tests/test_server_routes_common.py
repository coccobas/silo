"""Tests for common routes — health and models."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from silo.server.app import create_app


class MockBackend:
    def health(self) -> dict:
        return {"status": "ok", "model": "test-model", "backend": "mock"}

    def load(self, model_path: str, config: dict) -> None:
        pass

    def unload(self) -> None:
        pass

    def chat(self, messages, stream=False, **kwargs):
        return {"choices": [{"message": {"content": "hi"}}]}


@pytest.fixture
def app():
    return create_app(MockBackend(), "test-model")


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model"] == "test-model"
        assert data["backend"] == "mock"


@pytest.mark.asyncio
class TestModelsEndpoint:
    async def test_list_models(self, client):
        response = await client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "silo"
