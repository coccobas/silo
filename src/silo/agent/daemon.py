"""Agent daemon — FastAPI app that exposes local management over HTTP."""

from __future__ import annotations

import logging
import platform
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from silo.agent.schemas import (
    CheckResultResponse,
    DownloadRequest,
    DownloadResponse,
    MemoryInfoResponse,
    NodeStatusResponse,
    ProcessInfoResponse,
    RegistryEntryResponse,
    SpawnRequest,
    SpawnResponse,
    StopRequest,
    StopResponse,
)

logger = logging.getLogger(__name__)


def create_agent_app(
    node_name: str | None = None,
    port: int = 9900,
    head: bool = False,
) -> FastAPI:
    """Create the agent daemon FastAPI application.

    Args:
        node_name: Name to advertise via mDNS. If None, mDNS is skipped.
        port: Port number for mDNS advertisement.
        head: Enable head node mode with cluster coordination endpoints.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        advertiser = None
        if node_name is not None:
            try:
                from silo.agent.discovery import ServiceAdvertiser

                advertiser = ServiceAdvertiser(node_name=node_name, port=port)
                advertiser.__enter__()
            except ImportError:
                logger.debug(
                    "zeroconf not installed, skipping mDNS advertisement"
                )

        # Head mode: start cluster state and health checker
        if head and node_name:
            from silo.agent.client import RemoteClient
            from silo.agent.cluster import (
                ClusterState,
                HealthChecker,
                auto_discover_workers,
            )
            from silo.agent.cluster_schemas import HealthConfig
            from silo.agent.retry import RetryConfig
            from silo.config.models import NodeConfig

            config = HealthConfig()
            cluster = ClusterState(config)
            app.state.cluster = cluster
            app.state.head_name = node_name

            # Register self as a worker
            cluster.register_worker(node_name, "127.0.0.1", port)
            cluster.record_health_success(node_name)

            # Auto-discover existing workers via mDNS
            try:
                await auto_discover_workers(
                    cluster, timeout=2.0, exclude_name=node_name
                )
            except ImportError:
                logger.debug("mDNS discovery unavailable, skipping auto-discover")

            # Start health checker
            def client_factory(name: str, host: str, p: int) -> RemoteClient:
                return RemoteClient(
                    NodeConfig(name=name, host=host, port=p),
                    retry_config=RetryConfig(max_retries=1, base_delay=0.5, max_delay=2.0),
                )

            health_checker = HealthChecker(cluster, client_factory, config)
            await health_checker.start()
            app.state.health_checker = health_checker
            logger.info("Head node '%s' started with cluster coordination", node_name)

        yield

        # Shutdown
        if head and hasattr(app.state, "health_checker"):
            await app.state.health_checker.stop()
        if advertiser is not None:
            advertiser.__exit__(None, None, None)

    app = FastAPI(
        title="Silo Agent",
        description="Remote management daemon for Silo nodes.",
        lifespan=lifespan,
    )

    # Mount cluster router if head mode
    if head:
        from silo.agent.cluster_router import router as cluster_router

        app.include_router(cluster_router)

    # ── Status ────────────────────────────────────────

    @app.get("/status", response_model=NodeStatusResponse)
    def node_status() -> NodeStatusResponse:
        from silo.process.manager import list_running
        from silo.process.memory import get_memory_info
        from silo.registry.store import Registry

        processes = list_running()
        mem = get_memory_info()
        registry = Registry.load()

        return NodeStatusResponse(
            hostname=platform.node(),
            processes=[
                ProcessInfoResponse(
                    name=p.name,
                    pid=p.pid,
                    port=p.port,
                    repo_id=p.repo_id,
                    status=p.status,
                )
                for p in processes
            ],
            memory=MemoryInfoResponse(
                total_gb=mem.total_gb,
                available_gb=mem.available_gb,
                used_gb=mem.used_gb,
                pressure=mem.pressure,
                usage_percent=mem.usage_percent,
            ),
            registry=[
                RegistryEntryResponse(
                    repo_id=e.repo_id,
                    format=str(e.format),
                    local_path=e.local_path,
                    size_bytes=e.size_bytes,
                    downloaded_at=e.downloaded_at,
                    tags=list(e.tags),
                )
                for e in registry.list()
            ],
        )

    # ── Processes ─────────────────────────────────────

    @app.get("/processes", response_model=list[ProcessInfoResponse])
    def list_processes() -> list[ProcessInfoResponse]:
        from silo.process.manager import list_running

        return [
            ProcessInfoResponse(
                name=p.name,
                pid=p.pid,
                port=p.port,
                repo_id=p.repo_id,
                status=p.status,
            )
            for p in list_running()
        ]

    @app.post("/spawn", response_model=SpawnResponse)
    def spawn(req: SpawnRequest) -> SpawnResponse:
        from silo.process.manager import spawn_model

        pid = spawn_model(
            name=req.name,
            repo_id=req.repo_id,
            host=req.host,
            port=req.port,
            quantize=req.quantize,
            output=req.output,
        )
        return SpawnResponse(pid=pid, name=req.name)

    @app.post("/stop", response_model=StopResponse)
    def stop(req: StopRequest) -> StopResponse:
        from silo.process.manager import stop_model

        stopped = stop_model(name=req.name, grace_period=req.grace_period)
        return StopResponse(stopped=stopped, name=req.name)

    # ── Memory ────────────────────────────────────────

    @app.get("/memory", response_model=MemoryInfoResponse)
    def memory() -> MemoryInfoResponse:
        from silo.process.memory import get_memory_info

        mem = get_memory_info()
        return MemoryInfoResponse(
            total_gb=mem.total_gb,
            available_gb=mem.available_gb,
            used_gb=mem.used_gb,
            pressure=mem.pressure,
            usage_percent=mem.usage_percent,
        )

    # ── Registry ──────────────────────────────────────

    @app.get("/registry", response_model=list[RegistryEntryResponse])
    def registry() -> list[RegistryEntryResponse]:
        from silo.registry.store import Registry

        return [
            RegistryEntryResponse(
                repo_id=e.repo_id,
                format=str(e.format),
                local_path=e.local_path,
                size_bytes=e.size_bytes,
                downloaded_at=e.downloaded_at,
                tags=list(e.tags),
            )
            for e in Registry.load().list()
        ]

    # ── Download ──────────────────────────────────────

    @app.post("/download", response_model=DownloadResponse)
    def download(req: DownloadRequest) -> DownloadResponse:
        from silo.download.hf import download_model, get_model_info
        from silo.registry.detector import detect_model_format
        from silo.registry.models import RegistryEntry
        from silo.registry.store import Registry

        path = download_model(req.repo_id, local_dir=req.local_dir)
        info = get_model_info(req.repo_id)
        fmt = detect_model_format(req.repo_id, info.get("siblings"))
        entry = RegistryEntry(
            repo_id=req.repo_id,
            format=fmt,
            local_path=str(path),
        )
        reg = Registry.load()
        updated = reg.add(entry)
        updated.save()
        return DownloadResponse(repo_id=req.repo_id, local_path=str(path))

    # ── Doctor ────────────────────────────────────────

    @app.get("/doctor", response_model=list[CheckResultResponse])
    def doctor() -> list[CheckResultResponse]:
        from silo.doctor.checks import run_all_checks

        return [
            CheckResultResponse(
                name=c.name, status=c.status, message=c.message
            )
            for c in run_all_checks()
        ]

    # ── Health ────────────────────────────────────────

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "hostname": platform.node()}

    return app
