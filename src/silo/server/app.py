"""Server application factory."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from silo.backends.protocols import ChatBackend, SttBackend, TtsBackend
from silo.server.errors import runtime_error_handler, value_error_handler
from silo.server.metrics import ModelMetrics
from silo.server.routes_common import router as common_router
from silo.server.routes_metrics import router as metrics_router


def create_app(backend: Any, model_name: str) -> FastAPI:
    """Create a FastAPI app for serving a single model.

    Conditionally registers routes based on which protocol
    the backend implements (ChatBackend, SttBackend, TtsBackend).

    Args:
        backend: A backend instance implementing one or more Backend protocols.
        model_name: Friendly name for the model.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title=f"Silo — {model_name}",
        description="OpenAI-compatible local model server",
    )

    app.state.backend = backend
    app.state.model_name = model_name
    app.state.metrics = ModelMetrics(model_name=model_name)

    app.add_exception_handler(RuntimeError, runtime_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ValueError, value_error_handler)  # type: ignore[arg-type]

    # Always include common routes (health, models, metrics)
    app.include_router(common_router)
    app.include_router(metrics_router)

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
