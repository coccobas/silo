"""Server application factory."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI

from silo.backends.protocols import ChatBackend, SttBackend, TtsBackend
from silo.config.models import LitellmConfig
from silo.server.errors import runtime_error_handler, value_error_handler
from silo.server.metrics import ModelMetrics
from silo.server.routes_common import router as common_router
from silo.server.routes_metrics import router as metrics_router

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LitellmRegistration:
    """LiteLLM registration info attached to a model server."""

    config: LitellmConfig
    host: str
    port: int


@dataclass
class LitellmState:
    """Mutable LiteLLM state for a running model server.

    This is the single mutable point for LiteLLM integration.
    Admin endpoints read/update this; lifespan shutdown reads from it
    so that changes made at runtime are respected on exit.
    """

    registered: bool = False
    url: str = ""
    api_key: str = ""
    model_name: str = ""
    instance_id: str = ""
    host: str = ""
    port: int = 0


def create_app(
    backend: Any,
    model_name: str,
    litellm: LitellmRegistration | None = None,
) -> FastAPI:
    """Create a FastAPI app for serving a single model.

    Conditionally registers routes based on which protocol
    the backend implements (ChatBackend, SttBackend, TtsBackend).

    When *litellm* is provided, the server self-registers with the
    LiteLLM proxy on startup and deregisters on shutdown.

    Args:
        backend: A backend instance implementing one or more Backend protocols.
        model_name: Friendly name for the model.
        litellm: Optional LiteLLM registration info.

    Returns:
        Configured FastAPI application.
    """
    instance_id = str(uuid.uuid4())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        state: LitellmState = app.state.litellm_state

        # Register with LiteLLM on startup (server is ready to serve)
        if litellm and litellm.config.enabled:
            from silo.litellm.registry import register_model

            register_model(
                litellm.config, model_name,
                litellm.host, litellm.port, instance_id,
            )
            state.registered = True
            state.url = litellm.config.url
            state.api_key = litellm.config.api_key
            state.model_name = model_name
            state.instance_id = instance_id
            state.host = litellm.host
            state.port = litellm.port

        yield

        # Deregister on shutdown — read live state (may have been changed by admin)
        if state.registered:
            from silo.litellm.registry import deregister_model

            config = LitellmConfig(
                enabled=True, url=state.url, api_key=state.api_key,
            )
            deregister_model(config, state.model_name, state.instance_id)

    # Initialize mutable LiteLLM state
    litellm_state = LitellmState(
        model_name=model_name,
        instance_id=instance_id,
        host=litellm.host if litellm else "",
        port=litellm.port if litellm else 0,
    )

    app = FastAPI(
        title=f"Silo — {model_name}",
        description="OpenAI-compatible local model server",
        lifespan=lifespan,
    )

    app.state.backend = backend
    app.state.model_name = model_name
    app.state.instance_id = instance_id
    app.state.litellm_state = litellm_state
    app.state.metrics = ModelMetrics(model_name=model_name)

    app.add_exception_handler(RuntimeError, runtime_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ValueError, value_error_handler)  # type: ignore[arg-type]

    # Always include common routes (health, models, metrics)
    app.include_router(common_router)
    app.include_router(metrics_router)

    # Admin routes (LiteLLM management, server info)
    from silo.server.routes_admin import router as admin_router

    app.include_router(admin_router)

    # Conditionally include modality-specific routes
    if isinstance(backend, ChatBackend):
        from silo.server.routes_chat import router as chat_router

        app.include_router(chat_router)

    if isinstance(backend, SttBackend) or hasattr(backend, "transcribe"):
        from silo.server.routes_audio import router as audio_router

        app.include_router(audio_router)
    elif isinstance(backend, TtsBackend) or hasattr(backend, "speak"):
        from silo.server.routes_audio import router as audio_router

        app.include_router(audio_router)

    return app
