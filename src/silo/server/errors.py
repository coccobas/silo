"""OpenAI-compatible error responses."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL_NOT_FOUND = "model_not_found"
MODEL_LOADING = "model_loading"
INVALID_REQUEST = "invalid_request"
BACKEND_ERROR = "backend_error"
OUT_OF_MEMORY = "out_of_memory"
TIMEOUT = "timeout"


class OpenAIError(BaseModel):
    """OpenAI-compatible error body."""

    model_config = {"frozen": True}

    message: str
    type: str
    code: int


class OpenAIErrorResponse(BaseModel):
    """OpenAI-compatible error envelope."""

    model_config = {"frozen": True}

    error: OpenAIError


def openai_error_response(
    status_code: int,
    message: str,
    error_type: str,
) -> JSONResponse:
    """Create a JSONResponse matching OpenAI's error format."""
    body = OpenAIErrorResponse(
        error=OpenAIError(message=message, type=error_type, code=status_code)
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
    )


async def runtime_error_handler(_request: Request, exc: RuntimeError) -> JSONResponse:
    return openai_error_response(500, str(exc), BACKEND_ERROR)


async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    return openai_error_response(400, str(exc), INVALID_REQUEST)
