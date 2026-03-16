"""Request/response schemas for the cluster API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from silo.agent.schemas import MemoryInfoResponse, ProcessInfoResponse


# ── Cluster state models ─────────────────────────


class WorkerNode(BaseModel):
    """Snapshot of a worker node's state in the cluster."""

    model_config = {"frozen": True}

    name: str
    host: str
    port: int = 9900
    status: Literal["healthy", "unhealthy", "unknown"] = "unknown"
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    consecutive_failures: int = 0


class HealthConfig(BaseModel):
    """Configuration for the cluster health checker."""

    model_config = {"frozen": True}

    check_interval: float = 10.0
    failure_threshold: int = 3
    timeout: float = 5.0


# ── Request models ───────────────────────────────


class RegisterRequest(BaseModel):
    """Request to register a worker with the head node."""

    name: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = 9900


class ClusterSpawnRequest(BaseModel):
    """Request to spawn a model via the cluster head."""

    name: str
    repo_id: str
    host: str = "127.0.0.1"
    port: int = 8800
    quantize: str | None = None
    output: str | None = None
    node: str | None = None  # None = auto-place


class ClusterStopRequest(BaseModel):
    """Request to stop a model via the cluster head."""

    name: str
    grace_period: int = 30


# ── Response models ──────────────────────────────


class WorkerNodeResponse(BaseModel):
    """Worker node info in cluster status response."""

    name: str
    host: str
    port: int
    status: str
    processes: list[ProcessInfoResponse] = Field(default_factory=list)
    memory: MemoryInfoResponse | None = None


class ClusterStatusResponse(BaseModel):
    """Aggregated cluster status."""

    head: str
    workers: list[WorkerNodeResponse]
    total_models: int
    total_memory_gb: float
    total_available_gb: float


class ClusterSpawnResponse(BaseModel):
    """Response after cluster-level spawn."""

    node: str
    pid: int
    name: str


class ClusterStopResponse(BaseModel):
    """Response after cluster-level stop."""

    node: str
    stopped: bool
    name: str
