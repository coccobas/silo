"""Pydantic schemas for LiteLLM management API payloads."""

from __future__ import annotations

from pydantic import BaseModel


class LitellmModelParams(BaseModel):
    """Parameters describing how LiteLLM should reach a model."""

    model_config = {"frozen": True}

    model: str
    api_base: str
    api_key: str = "unused"


class LitellmRegisterPayload(BaseModel):
    """POST /model/new request body."""

    model_config = {"frozen": True}

    model_name: str
    litellm_params: LitellmModelParams
    model_info: dict[str, str] = {}


class LitellmDeletePayload(BaseModel):
    """POST /model/delete request body."""

    model_config = {"frozen": True}

    id: str
