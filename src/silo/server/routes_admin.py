"""Admin routes — LiteLLM management and server info for running servers."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

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
    the old entry first.  Uses the LiteLLM client directly to avoid
    slow subprocess calls (like netbird status) that block the server.
    """
    from silo.litellm.client import LitellmClient
    from silo.litellm.registry import normalize_litellm_url

    state = request.app.state.litellm_state
    url = normalize_litellm_url(req.url)
    api_key = req.api_key
    new_name = req.model_name or state.model_name or request.app.state.model_name

    client = LitellmClient(url, api_key)

    # Deregister old if currently registered
    if state.registered and state.instance_id:
        old_client = LitellmClient(state.url, state.api_key)
        old_client.delete(state.instance_id)

    # Build api_base from the server's own host:port
    # Use the request's client host (how the caller reached us) for
    # externally-reachable address, falling back to state.host
    client_host = request.client.host if request.client else state.host
    serve_host = client_host if client_host not in ("127.0.0.1", "::1") else state.host
    api_base = f"http://{serve_host}:{state.port}/v1"

    client.register(new_name, api_base, state.instance_id)

    # Update live state
    state.registered = True
    state.url = url
    state.api_key = api_key
    state.model_name = new_name

    logger.info("Admin: registered '%s' with LiteLLM at %s (api_base=%s)", new_name, url, api_base)
    return _litellm_status(request)


@router.post("/litellm/deregister", response_model=LitellmStatusResponse)
def litellm_deregister(request: Request) -> LitellmStatusResponse:
    """Deregister this server from LiteLLM."""
    from silo.litellm.client import LitellmClient

    state = request.app.state.litellm_state

    if state.registered and state.instance_id:
        client = LitellmClient(state.url, state.api_key)
        client.delete(state.instance_id)
        state.registered = False
        logger.info("Admin: deregistered '%s' from LiteLLM", state.model_name)

    return _litellm_status(request)


@router.put("/model-name", response_model=ServerInfoResponse)
def update_model_name(
    req: ModelNameRequest, request: Request,
) -> ServerInfoResponse:
    """Change the model name (re-registers with LiteLLM if registered)."""
    from silo.litellm.client import LitellmClient

    state = request.app.state.litellm_state
    old_name = request.app.state.model_name

    # Update the server's model name
    request.app.state.model_name = req.model_name

    # Re-register with LiteLLM under the new name
    if state.registered and state.instance_id:
        client = LitellmClient(state.url, state.api_key)
        client.delete(state.instance_id)

        client_host = request.client.host if request.client else state.host
        serve_host = client_host if client_host not in ("127.0.0.1", "::1") else state.host
        api_base = f"http://{serve_host}:{state.port}/v1"
        client.register(req.model_name, api_base, state.instance_id)

        state.model_name = req.model_name
        logger.info("Admin: renamed '%s' -> '%s' on LiteLLM", old_name, req.model_name)
    else:
        state.model_name = req.model_name

    return ServerInfoResponse(
        model_name=request.app.state.model_name,
        instance_id=request.app.state.instance_id,
        litellm=_litellm_status(request),
    )
