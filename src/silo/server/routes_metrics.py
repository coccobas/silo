"""Metrics route — Prometheus-compatible /metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

router = APIRouter()


@router.get("/metrics")
async def metrics(request: Request) -> PlainTextResponse:
    """Prometheus-compatible metrics endpoint."""
    model_metrics = getattr(request.app.state, "metrics", None)
    if model_metrics is None:
        return PlainTextResponse("# No metrics available\n")
    return PlainTextResponse(
        model_metrics.to_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
