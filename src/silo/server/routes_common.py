"""Common routes — health and models list."""

from __future__ import annotations

from fastapi import APIRouter, Request

from silo.server.schemas import HealthResponse, ModelListResponse, ModelObject

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    backend = request.app.state.backend
    health_data = backend.health()
    return HealthResponse(
        status=health_data.get("status", "unknown"),
        model=health_data.get("model", ""),
        backend=health_data.get("backend", "unknown"),
    )


@router.get("/v1/models")
async def list_models(request: Request) -> ModelListResponse:
    """List served models (single model per instance)."""
    model_name = request.app.state.model_name
    return ModelListResponse(
        data=[ModelObject(id=model_name)],
    )
