"""Admin routes — LiteLLM management and server info for running servers."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from silo.config.models import LitellmConfig
from silo.server.admin_schemas import (
    LitellmRegisterRequest,
    LitellmStatusResponse,
    ModelNameRequest,
    ServerInfoResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def _litellm_status(request: Request) -> LitellmStatusResponse:
    """Build a LiteLLM status response from current state."""
    state = request.app.state.litellm_state
    return LitellmStatusResponse(
        registered=state.registered,
        url=state.url,
        model_name=state.model_name,
        instance_id=state.instance_id,
    )


@router.get("/info", response_model=ServerInfoResponse)
def server_info(request: Request) -> ServerInfoResponse:
    """Return current model server state."""
    return ServerInfoResponse(
        model_name=request.app.state.model_name,
        instance_id=request.app.state.instance_id,
        litellm=_litellm_status(request),
    )


@router.post("/litellm/register", response_model=LitellmStatusResponse)
def litellm_register(
    req: LitellmRegisterRequest, request: Request,
) -> LitellmStatusResponse:
    """Register (or re-register) this server with a LiteLLM proxy.

    If already registered with a different URL or name, deregisters
    the old entry first.
    """
    from silo.litellm.registry import (
        deregister_model,
        normalize_litellm_url,
        register_model,
    )

    state = request.app.state.litellm_state
    url = normalize_litellm_url(req.url)
    new_name = req.model_name or state.model_name or request.app.state.model_name

    # Deregister old if currently registered
    if state.registered and state.instance_id:
        old_config = LitellmConfig(
            enabled=True, url=state.url, api_key=state.api_key,
        )
        deregister_model(old_config, state.model_name, state.instance_id)

    # Register with new settings
    config = LitellmConfig(enabled=True, url=url, api_key=req.api_key)
    register_model(config, new_name, state.host, state.port, state.instance_id)

    # Update live state
    state.registered = True
    state.url = url
    state.api_key = req.api_key
    state.model_name = new_name

    logger.info("Admin: registered '%s' with LiteLLM at %s", new_name, url)
    return _litellm_status(request)


@router.post("/litellm/deregister", response_model=LitellmStatusResponse)
def litellm_deregister(request: Request) -> LitellmStatusResponse:
    """Deregister this server from LiteLLM."""
    from silo.litellm.registry import deregister_model

    state = request.app.state.litellm_state

    if state.registered and state.instance_id:
        config = LitellmConfig(
            enabled=True, url=state.url, api_key=state.api_key,
        )
        deregister_model(config, state.model_name, state.instance_id)
        state.registered = False
        logger.info("Admin: deregistered '%s' from LiteLLM", state.model_name)

    return _litellm_status(request)


@router.put("/model-name", response_model=ServerInfoResponse)
def update_model_name(
    req: ModelNameRequest, request: Request,
) -> ServerInfoResponse:
    """Change the model name (re-registers with LiteLLM if registered)."""
    from silo.litellm.registry import deregister_model, register_model

    state = request.app.state.litellm_state
    old_name = request.app.state.model_name

    # Update the server's model name
    request.app.state.model_name = req.model_name

    # Re-register with LiteLLM under the new name
    if state.registered and state.instance_id:
        config = LitellmConfig(
            enabled=True, url=state.url, api_key=state.api_key,
        )
        deregister_model(config, state.model_name, state.instance_id)
        register_model(config, req.model_name, state.host, state.port, state.instance_id)
        state.model_name = req.model_name
        logger.info("Admin: renamed '%s' -> '%s' on LiteLLM", old_name, req.model_name)
    else:
        state.model_name = req.model_name

    return ServerInfoResponse(
        model_name=request.app.state.model_name,
        instance_id=request.app.state.instance_id,
        litellm=_litellm_status(request),
    )
