"""Tests for metrics collection and endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from silo.server.app import create_app
from silo.server.metrics import ModelMetrics


class TestModelMetrics:
    def test_empty_metrics(self):
        m = ModelMetrics(model_name="test")
        output = m.to_prometheus()
        assert "silo_errors_total" in output
        assert "silo_tokens_total" in output
        assert "silo_tokens_per_second" in output

    def test_record_request(self):
        m = ModelMetrics(model_name="test")
        m.record_request("/v1/chat/completions", 200, 0.5)
        m.record_request("/v1/chat/completions", 200, 1.0)
        m.record_request("/v1/chat/completions", 500, 0.1)
        output = m.to_prometheus()
        assert 'endpoint="/v1/chat/completions",status="200"} 2' in output
        assert 'endpoint="/v1/chat/completions",status="500"} 1' in output
        assert 'silo_errors_total{model="test"} 1' in output

    def test_record_tokens(self):
        m = ModelMetrics(model_name="test")
        m.record_tokens(100)
        m.record_tokens(50)
        output = m.to_prometheus()
        assert 'silo_tokens_total{model="test"} 150' in output

    def test_tokens_per_second(self):
        m = ModelMetrics(model_name="test")
        m.record_tokens(100)
        m.record_duration(2.0)
        output = m.to_prometheus()
        assert 'silo_tokens_per_second{model="test"} 50.0' in output

    def test_duration_quantiles(self):
        m = ModelMetrics(model_name="test")
        for i in range(100):
            m.record_request("/v1/chat/completions", 200, float(i) / 100)
        output = m.to_prometheus()
        assert 'quantile="0.5"' in output
        assert 'quantile="0.95"' in output
        assert 'quantile="0.99"' in output

    def test_model_loaded(self):
        m = ModelMetrics(model_name="test")
        output = m.to_prometheus()
        assert 'silo_model_loaded{model="test"} 0' in output

        m.record_request("/health", 200, 0.01)
        output = m.to_prometheus()
        assert 'silo_model_loaded{model="test"} 1' in output


class MockBackend:
    def health(self):
        return {"status": "ok", "model": "test", "backend": "mock"}

    def load(self, model_path, config):
        pass

    def unload(self):
        pass

    def chat(self, messages, stream=False, **kwargs):
        return {"choices": [{"message": {"content": "hi"}}]}


@pytest.mark.asyncio
class TestMetricsEndpoint:
    async def test_metrics_endpoint(self):
        app = create_app(MockBackend(), "test-model")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics")
        assert response.status_code == 200
        assert "silo_" in response.text
        assert "text/plain" in response.headers["content-type"]

    async def test_metrics_after_requests(self):
        app = create_app(MockBackend(), "test-model")
        # Record some metrics
        app.state.metrics.record_request("/v1/chat/completions", 200, 0.5)
        app.state.metrics.record_tokens(42)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics")
        assert response.status_code == 200
        assert "silo_tokens_total" in response.text
        assert "42" in response.text
