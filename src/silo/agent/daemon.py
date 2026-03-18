"""Agent daemon — FastAPI app that exposes local management over HTTP."""

from __future__ import annotations

import logging
import platform
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

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
    SystemStatsResponse,
)

logger = logging.getLogger(__name__)


def _detect_local_ip() -> str:
    """Detect the machine's primary non-localhost IP address.

    Checks all interfaces to find VPN IPs (Netbird/WireGuard use 100.x.x.x)
    alongside regular LAN IPs. Prefers VPN IPs for cluster display.
    """
    import socket

    ips: list[str] = []

    # Method 1: default route IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass

    # Method 2: all IPs from hostname resolution
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass

    # Method 3: try common VPN interface names
    try:
        for iface in ("wt0", "netbird0", "utun", "wg0"):
            try:
                for info in socket.getaddrinfo(iface, None, socket.AF_INET):
                    ip = info[4][0]
                    if not ip.startswith("127.") and ip not in ips:
                        ips.append(ip)
            except Exception:
                continue
    except Exception:
        pass

    if not ips:
        return "127.0.0.1"

    # Prefer VPN IPs (100.x.x.x for Netbird, 10.x.x.x for others)
    for ip in ips:
        if ip.startswith("100.") or ip.startswith("10."):
            return ip
    return ips[0]


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
        import asyncio

        from silo.config.loader import load_config

        loop = asyncio.get_running_loop()

        # Load LiteLLM config for optional deregister-all on quit
        silo_config = load_config()
        app.state.litellm_config = silo_config.litellm

        advertiser = None
        if node_name is not None:
            try:
                from silo.agent.discovery import ServiceAdvertiser

                role = "head" if head else "worker"
                advertiser = ServiceAdvertiser(
                    node_name=node_name, port=port, role=role
                )
                # Zeroconf does blocking I/O — run in thread
                await loop.run_in_executor(None, advertiser.__enter__)
            except ImportError:
                logger.debug(
                    "zeroconf not installed, skipping mDNS advertisement"
                )
            except Exception:
                logger.warning("mDNS advertisement failed", exc_info=True)
                advertiser = None

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

            from silo.config.paths import CLUSTER_WORKERS_PATH

            config = HealthConfig()
            cluster = ClusterState(
                config, persist_path=str(CLUSTER_WORKERS_PATH)
            )
            app.state.cluster = cluster
            app.state.head_name = node_name

            # Load previously registered workers
            cluster.load_persisted()

            # Register self as a worker
            from silo import __version__

            local_ip = _detect_local_ip()
            cluster.register_worker(node_name, local_ip, port)
            cluster.record_health_success(node_name, version=__version__)

            # Auto-discover existing workers via mDNS
            try:
                await auto_discover_workers(
                    cluster, timeout=2.0, exclude_name=node_name
                )
            except ImportError:
                logger.debug("mDNS discovery unavailable, skipping auto-discover")
            except Exception:
                logger.warning("mDNS auto-discovery failed", exc_info=True)

            # Start health checker
            def client_factory(name: str, host: str, p: int) -> RemoteClient:
                return RemoteClient(
                    NodeConfig(name=name, host=host, port=p),
                    retry_config=RetryConfig(max_retries=1, base_delay=0.5, max_delay=2.0),
                )

            health_checker = HealthChecker(
                cluster, client_factory, config,
                exclude_name=node_name,
                head_url=f"http://0.0.0.0:{port}",
            )
            await health_checker.start()
            app.state.health_checker = health_checker
            logger.info("Head node '%s' started with cluster coordination", node_name)

        yield

        # Shutdown
        if head and hasattr(app.state, "health_checker"):
            await app.state.health_checker.stop()

        # Optionally deregister all models from LiteLLM on quit
        if hasattr(app.state, "litellm_config") and app.state.litellm_config.deregister_on_quit:
            from silo.litellm.registry import deregister_all

            local_ip = _detect_local_ip()
            deregister_all(app.state.litellm_config, api_base_prefix=f"http://{local_ip}")

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
        from silo.process.system_stats import get_system_stats
        from silo.registry.store import Registry

        processes = list_running()
        mem = get_memory_info()
        stats = get_system_stats()
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
            system_stats=SystemStatsResponse(
                cpu_percent=stats.cpu_percent,
                gpu_percent=stats.gpu_percent,
                gpu_name=stats.gpu_name,
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

        result = spawn_model(
            name=req.name,
            repo_id=req.repo_id,
            host=req.host,
            port=req.port,
            quantize=req.quantize,
            output=req.output,
        )
        return SpawnResponse(pid=result.pid, name=req.name)

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

    # ── System Stats ─────────────────────────────────

    @app.get("/system-stats", response_model=SystemStatsResponse)
    def system_stats() -> SystemStatsResponse:
        from silo.process.system_stats import get_system_stats

        stats = get_system_stats()
        return SystemStatsResponse(
            cpu_percent=stats.cpu_percent,
            gpu_percent=stats.gpu_percent,
            gpu_name=stats.gpu_name,
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
        from silo import __version__

        return {"status": "ok", "hostname": platform.node(), "version": __version__}

    # ── Head announcement ─────────────────────────────

    app.state.head_url = None

    @app.post("/announce-head")
    async def announce_head(request: Request) -> dict[str, str]:
        """Called by the head to tell this worker where the head is."""
        body = await request.json()
        url = body.get("url")
        if url:
            app.state.head_url = url
            logger.info("Head node announced at %s", url)
        return {"status": "ok"}

    @app.get("/head")
    def get_head() -> dict[str, str | None]:
        """Return the head node URL if known."""
        return {"head_url": app.state.head_url}

    return app
