"""Request/response schemas for the model server admin API."""

from __future__ import annotations

from pydantic import BaseModel


class LitellmRegisterRequest(BaseModel):
    """Request to register (or re-register) with a LiteLLM proxy."""

    url: str
    api_key: str = ""
    model_name: str | None = None


class LitellmStatusResponse(BaseModel):
    """Current LiteLLM registration status."""

    registered: bool
    url: str
    model_name: str
    instance_id: str


class ServerInfoResponse(BaseModel):
    """Current model server state."""

    model_name: str
    instance_id: str
    litellm: LitellmStatusResponse


class ModelNameRequest(BaseModel):
    """Request to change the model name."""

    model_name: str
