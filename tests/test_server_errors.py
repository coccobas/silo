"""Tests for OpenAI-compatible error responses."""

from silo.server.errors import (
    BACKEND_ERROR,
    INVALID_REQUEST,
    MODEL_NOT_FOUND,
    OpenAIError,
    OpenAIErrorResponse,
    openai_error_response,
)


class TestOpenAIError:
    def test_error_model(self):
        err = OpenAIError(message="not found", type=MODEL_NOT_FOUND, code=404)
        assert err.message == "not found"
        assert err.type == "model_not_found"
        assert err.code == 404

    def test_error_response_model(self):
        err = OpenAIError(message="bad request", type=INVALID_REQUEST, code=400)
        resp = OpenAIErrorResponse(error=err)
        data = resp.model_dump()
        assert data["error"]["message"] == "bad request"
        assert data["error"]["type"] == "invalid_request"


class TestOpenaiErrorResponse:
    def test_json_response(self):
        resp = openai_error_response(500, "server error", BACKEND_ERROR)
        assert resp.status_code == 500
        assert resp.body is not None

    def test_404_response(self):
        resp = openai_error_response(404, "model not found", MODEL_NOT_FOUND)
        assert resp.status_code == 404
