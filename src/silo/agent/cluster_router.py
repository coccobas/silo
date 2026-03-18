"""FastAPI router for cluster-level endpoints.

Mounted on the daemon only when head mode is enabled.
All cluster state access goes through app.state.cluster.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from silo.agent.client import LocalClient, RemoteClient
from silo.agent.cluster import ClusterState, select_node
from silo.agent.cluster_schemas import (
    ClusterDownloadRequest,
    ClusterDownloadResponse,
    ClusterSpawnRequest,
    ClusterSpawnResponse,
    ClusterStatusResponse,
    ClusterStopRequest,
    ClusterStopResponse,
    RegisterRequest,
    WorkerNodeResponse,
)
from silo.agent.retry import RetryConfig
from silo.agent.schemas import (
    MemoryInfoResponse,
    ProcessInfoResponse,
    SystemStatsResponse,
)
from silo.config.models import NodeConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cluster", tags=["cluster"])


def _get_cluster(request: Request) -> ClusterState:
    return request.app.state.cluster


def _get_head_name(request: Request) -> str:
    return request.app.state.head_name


def _build_client(worker_name: str, host: str, port: int) -> LocalClient | RemoteClient:
    """Build a client for a worker — local if it's the head, remote otherwise."""
    # This is called within request context; head_name is on app.state
    return RemoteClient(
        NodeConfig(name=worker_name, host=host, port=port),
        retry_config=RetryConfig(max_retries=2, base_delay=0.3, max_delay=3.0),
    )


# ── Registration ─────────────────────────────────


@router.post("/register", response_model=WorkerNodeResponse)
def register_worker(req: RegisterRequest, request: Request) -> WorkerNodeResponse:
    """Register a worker node with the cluster head."""
    cluster = _get_cluster(request)
    worker = cluster.register_worker(req.name, req.host, req.port)
    return WorkerNodeResponse(
        name=worker.name,
        host=worker.host,
        port=worker.port,
        status=worker.status,
    )


@router.delete("/workers/{worker_name}")
def unregister_worker(worker_name: str, request: Request) -> dict[str, str]:
    """Remove a worker node from the cluster."""
    cluster = _get_cluster(request)
    head_name = _get_head_name(request)
    if worker_name == head_name:
        raise HTTPException(
            status_code=400, detail="Cannot remove the head node"
        )
    if cluster.unregister_worker(worker_name):
        return {"removed": worker_name}
    raise HTTPException(
        status_code=404, detail=f"Worker '{worker_name}' not found"
    )


# ── Status ───────────────────────────────────────


@router.get("/status", response_model=ClusterStatusResponse)
def cluster_status(request: Request) -> ClusterStatusResponse:
    """Aggregated status of all cluster nodes."""
    cluster = _get_cluster(request)
    head_name = _get_head_name(request)
    workers = cluster.get_workers()

    worker_responses: list[WorkerNodeResponse] = []
    total_models = 0
    total_memory = 0.0
    total_available = 0.0

    for worker in workers:
        processes: list[ProcessInfoResponse] = []
        memory: MemoryInfoResponse | None = None
        sys_stats: SystemStatsResponse | None = None

        # Only query workers that are healthy or are the head itself
        # to avoid blocking on unreachable workers
        if worker.status == "healthy" or worker.name == head_name:
            try:
                if worker.name == head_name:
                    client = LocalClient()
                else:
                    client = _build_client(worker.name, worker.host, worker.port)
                procs = client.list_processes()
                processes = [
                    ProcessInfoResponse(
                        name=p.name, pid=p.pid, port=p.port,
                        repo_id=p.repo_id, status=p.status,
                    )
                    for p in procs
                ]
                mem = client.memory()
                memory = MemoryInfoResponse(
                    total_gb=mem.total_gb,
                    available_gb=mem.available_gb,
                    used_gb=mem.used_gb,
                    pressure=mem.pressure,
                    usage_percent=mem.usage_percent,
                )
                total_models += len(processes)
                total_memory += mem.total_gb
                total_available += mem.available_gb
                stats = client.system_stats()
                sys_stats = SystemStatsResponse(
                    cpu_percent=stats.cpu_percent,
                    gpu_percent=stats.gpu_percent,
                    gpu_name=stats.gpu_name,
                )
            except Exception:
                logger.warning(
                    "Could not query worker '%s' at %s:%d",
                    worker.name,
                    worker.host,
                    worker.port,
                )

        worker_responses.append(
            WorkerNodeResponse(
                name=worker.name,
                host=worker.host,
                port=worker.port,
                status=worker.status,
                version=worker.version,
                processes=processes,
                memory=memory,
                system_stats=sys_stats,
            )
        )

    return ClusterStatusResponse(
        head=head_name,
        workers=worker_responses,
        total_models=total_models,
        total_memory_gb=total_memory,
        total_available_gb=total_available,
    )


# ── Spawn ────────────────────────────────────────


@router.post("/spawn", response_model=ClusterSpawnResponse)
async def cluster_spawn(
    req: ClusterSpawnRequest, request: Request
) -> ClusterSpawnResponse:
    """Spawn a model on the best available node."""
    cluster = _get_cluster(request)
    head_name = _get_head_name(request)

    # Build clients for all workers
    clients: dict[str, LocalClient | RemoteClient] = {}
    for worker in cluster.get_workers():
        if worker.name == head_name:
            clients[worker.name] = LocalClient()
        else:
            clients[worker.name] = _build_client(
                worker.name, worker.host, worker.port
            )

    try:
        node_name, client = await select_node(
            cluster, clients, preferred_node=req.node
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    pid = client.spawn(  # type: ignore[union-attr]
        name=req.name,
        repo_id=req.repo_id,
        host=req.host,
        port=req.port,
        quantize=req.quantize,
        output=req.output,
    )
    logger.info("Spawned '%s' on node '%s' (pid=%d)", req.name, node_name, pid)

    # Register with LiteLLM — use the worker's actual host for the api_base
    if hasattr(request.app.state, "litellm_config"):
        from silo.litellm.registry import register_model
        from silo.process.pid import read_pid_entry

        worker = cluster.get_worker(node_name)
        worker_host = worker.host if worker else req.host
        entry = read_pid_entry(req.name)
        if entry and entry.instance_id:
            register_model(
                request.app.state.litellm_config, req.name,
                worker_host, req.port, entry.instance_id,
            )

    return ClusterSpawnResponse(node=node_name, pid=pid, name=req.name)


# ── Stop ─────────────────────────────────────────


@router.post("/stop", response_model=ClusterStopResponse)
def cluster_stop(req: ClusterStopRequest, request: Request) -> ClusterStopResponse:
    """Stop a model, searching across all cluster nodes."""
    cluster = _get_cluster(request)
    head_name = _get_head_name(request)

    # Read instance_id before stopping for LiteLLM deregistration
    from silo.process.pid import read_pid_entry

    entry = read_pid_entry(req.name)

    for worker in cluster.get_workers():
        try:
            if worker.name == head_name:
                client: LocalClient | RemoteClient = LocalClient()
            else:
                client = _build_client(worker.name, worker.host, worker.port)

            procs = client.list_processes()
            for proc in procs:
                if proc.name == req.name:
                    stopped = client.stop(
                        req.name, grace_period=req.grace_period
                    )

                    # Deregister from LiteLLM
                    if stopped and entry and hasattr(request.app.state, "litellm_config"):
                        from silo.litellm.registry import deregister_model

                        deregister_model(
                            request.app.state.litellm_config,
                            req.name, entry.instance_id,
                        )

                    return ClusterStopResponse(
                        node=worker.name, stopped=stopped, name=req.name
                    )
        except Exception:
            logger.debug(
                "Could not query worker '%s' while searching for '%s'",
                worker.name,
                req.name,
            )

    raise HTTPException(
        status_code=404,
        detail=f"Model '{req.name}' not found on any cluster node",
    )


# ── Download ────────────────────────────────────


@router.post("/download", response_model=ClusterDownloadResponse)
def cluster_download(
    req: ClusterDownloadRequest, request: Request
) -> ClusterDownloadResponse:
    """Download a model on a specific worker node."""
    cluster = _get_cluster(request)
    head_name = _get_head_name(request)

    worker = cluster.get_worker(req.node)
    if worker is None:
        raise HTTPException(
            status_code=404, detail=f"Node '{req.node}' not found in cluster"
        )

    if worker.name == head_name:
        client: LocalClient | RemoteClient = LocalClient()
    else:
        client = _build_client(worker.name, worker.host, worker.port)

    try:
        local_path = client.download(req.repo_id, local_dir=req.local_dir)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Download failed on '{req.node}': {exc}",
        ) from exc

    logger.info("Downloaded '%s' on node '%s'", req.repo_id, req.node)
    return ClusterDownloadResponse(
        node=req.node, repo_id=req.repo_id, local_path=local_path
    )
