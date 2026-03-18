"""Request/response schemas for the agent API."""

from __future__ import annotations

from pydantic import BaseModel


class SpawnRequest(BaseModel):
    """Request to start a model server."""

    name: str
    repo_id: str
    host: str = "127.0.0.1"
    port: int = 8800
    quantize: str | None = None
    output: str | None = None


class StopRequest(BaseModel):
    """Request to stop a model server."""

    name: str
    grace_period: int = 30


class DownloadRequest(BaseModel):
    """Request to download a model."""

    repo_id: str
    local_dir: str | None = None


class ProcessInfoResponse(BaseModel):
    """Info about a running process."""

    name: str
    pid: int
    port: int
    repo_id: str
    status: str


class MemoryInfoResponse(BaseModel):
    """System memory information."""

    total_gb: float
    available_gb: float
    used_gb: float
    pressure: str
    usage_percent: float


class CheckResultResponse(BaseModel):
    """Result of a diagnostic check."""

    name: str
    status: str
    message: str


class RegistryEntryResponse(BaseModel):
    """A model in the local registry."""

    repo_id: str
    format: str
    local_path: str | None = None
    size_bytes: int | None = None
    downloaded_at: str | None = None
    tags: list[str] = []


class SystemStatsResponse(BaseModel):
    """CPU and GPU usage information."""

    cpu_percent: float
    gpu_percent: float
    gpu_name: str


class NodeStatusResponse(BaseModel):
    """Full status of a node."""

    hostname: str
    processes: list[ProcessInfoResponse]
    memory: MemoryInfoResponse
    system_stats: SystemStatsResponse | None = None
    registry: list[RegistryEntryResponse]


class SpawnResponse(BaseModel):
    """Response after spawning a model."""

    pid: int
    name: str


class StopResponse(BaseModel):
    """Response after stopping a model."""

    stopped: bool
    name: str


class DownloadResponse(BaseModel):
    """Response after downloading a model."""

    repo_id: str
    local_path: str


class UpdateRequest(BaseModel):
    """Request to update a running model server."""

    name: str
    litellm_enabled: bool | None = None
    litellm_url: str | None = None
    litellm_api_key: str | None = None
    litellm_model_name: str | None = None
    model_name: str | None = None
    port: int | None = None


class UpdateResponse(BaseModel):
    """Response after updating a model server."""

    name: str
    restarted: bool
    changes: list[str]
