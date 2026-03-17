"""Cluster state management, placement, and health checking.

The head node uses these components to coordinate a cluster of
Silo agent workers. ClusterState is the central registry;
HealthChecker runs periodic background checks; select_node
implements memory-based placement.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from silo.agent.cluster_schemas import HealthConfig, WorkerNode

logger = logging.getLogger(__name__)


class ClusterState:
    """Mutable registry of worker nodes.

    Internal state is mutable, but all public accessors return frozen
    WorkerNode snapshots. Thread-safe via asyncio.Lock for use with
    the health checker background task.

    When *persist_path* is set, the worker list is saved to disk on
    every register/unregister so workers survive head restarts.
    """

    def __init__(
        self, config: HealthConfig, persist_path: str | None = None
    ) -> None:
        self._config = config
        self._workers: dict[str, WorkerNode] = {}
        self._lock = asyncio.Lock()
        self._persist_path = persist_path

    def load_persisted(self) -> int:
        """Load previously registered workers from disk.

        Returns the number of workers loaded. Workers start with
        status 'unknown' — the health checker will probe them.
        """
        if self._persist_path is None:
            return 0

        import json
        from pathlib import Path

        path = Path(self._persist_path)
        if not path.exists():
            return 0

        try:
            data = json.loads(path.read_text())
        except Exception:
            logger.warning("Could not read persisted workers from %s", path)
            return 0

        count = 0
        for entry in data:
            name = entry.get("name")
            host = entry.get("host")
            port = entry.get("port", 9900)
            if name and host and name not in self._workers:
                self._workers[name] = WorkerNode(
                    name=name, host=host, port=port, status="unknown"
                )
                count += 1
        if count:
            logger.info("Loaded %d persisted worker(s) from %s", count, path)
        return count

    def _persist(self) -> None:
        """Save current workers to disk."""
        if self._persist_path is None:
            return

        import json
        from pathlib import Path

        entries = [
            {"name": w.name, "host": w.host, "port": w.port}
            for w in self._workers.values()
        ]
        try:
            Path(self._persist_path).write_text(
                json.dumps(entries, indent=2)
            )
        except Exception:
            logger.warning("Could not persist workers to %s", self._persist_path)

    def register_worker(self, name: str, host: str, port: int) -> WorkerNode:
        """Register or update a worker node. Returns frozen snapshot."""
        existing = self._workers.get(name)
        worker = WorkerNode(
            name=name,
            host=host,
            port=port,
            status=existing.status if existing else "unknown",
            last_seen=existing.last_seen if existing else datetime.now(UTC),
            consecutive_failures=(
                existing.consecutive_failures if existing else 0
            ),
        )
        self._workers[name] = worker
        self._persist()
        logger.info("Registered worker '%s' at %s:%d", name, host, port)
        return worker

    def unregister_worker(self, name: str) -> bool:
        """Remove a worker. Returns True if it existed."""
        if name in self._workers:
            del self._workers[name]
            self._persist()
            logger.info("Unregistered worker '%s'", name)
            return True
        return False

    def get_worker(self, name: str) -> WorkerNode | None:
        """Get a frozen snapshot of a worker, or None."""
        return self._workers.get(name)

    def get_workers(self) -> tuple[WorkerNode, ...]:
        """Get frozen snapshots of all workers."""
        return tuple(self._workers.values())

    def get_healthy_workers(self) -> tuple[WorkerNode, ...]:
        """Get only workers with status 'healthy'."""
        return tuple(w for w in self._workers.values() if w.status == "healthy")

    def record_health_success(
        self, name: str, version: str | None = None
    ) -> WorkerNode:
        """Mark a worker as healthy, reset failure count.

        Raises KeyError if worker not registered.
        """
        existing = self._workers[name]
        updated = WorkerNode(
            name=existing.name,
            host=existing.host,
            port=existing.port,
            status="healthy",
            last_seen=datetime.now(UTC),
            consecutive_failures=0,
            version=version or existing.version,
        )
        self._workers[name] = updated
        return updated

    def record_health_failure(self, name: str) -> WorkerNode:
        """Record a health check failure.

        Marks unhealthy when consecutive failures reach the threshold.
        Raises KeyError if worker not registered.
        """
        existing = self._workers[name]
        failures = existing.consecutive_failures + 1
        status = (
            "unhealthy"
            if failures >= self._config.failure_threshold
            else existing.status
        )
        updated = WorkerNode(
            name=existing.name,
            host=existing.host,
            port=existing.port,
            status=status,
            last_seen=existing.last_seen,
            consecutive_failures=failures,
        )
        self._workers[name] = updated
        if status == "unhealthy" and existing.status != "unhealthy":
            logger.warning(
                "Worker '%s' marked unhealthy after %d failures",
                name,
                failures,
            )
        return updated


# ── Placement ────────────────────────────────────


async def select_node(
    cluster: ClusterState,
    clients: dict[str, object],
    preferred_node: str | None = None,
) -> tuple[str, object]:
    """Pick the best node for spawning a model.

    If *preferred_node* is set, validates it exists and is healthy.
    Otherwise queries memory on all healthy workers and picks the
    one with the most available memory.

    Returns (node_name, client).
    """
    if preferred_node is not None:
        worker = cluster.get_worker(preferred_node)
        if worker is None:
            raise ValueError(f"Node '{preferred_node}' not found in cluster")
        if worker.status == "unhealthy":
            raise ValueError(
                f"Node '{preferred_node}' is unhealthy "
                f"({worker.consecutive_failures} consecutive failures)"
            )
        client = clients.get(preferred_node)
        if client is None:
            raise ValueError(f"No client for node '{preferred_node}'")
        return preferred_node, client

    healthy = cluster.get_healthy_workers()
    if not healthy:
        raise RuntimeError("No healthy workers available in the cluster")

    # Query memory on all healthy workers
    candidates: list[tuple[str, object, float]] = []
    for worker in healthy:
        client = clients.get(worker.name)
        if client is None:
            continue
        try:
            mem = client.memory()  # type: ignore[union-attr]
            candidates.append((worker.name, client, mem.available_gb))
        except Exception:
            logger.debug("Could not query memory on '%s', skipping", worker.name)

    if not candidates:
        raise RuntimeError(
            "No reachable workers with available memory information"
        )

    # Pick the node with the most available memory
    candidates.sort(key=lambda c: c[2], reverse=True)
    best_name, best_client, best_mem = candidates[0]
    logger.info(
        "Selected node '%s' for spawn (%.1f GB available)", best_name, best_mem
    )
    return best_name, best_client


# ── Health checker ───────────────────────────────


class HealthChecker:
    """Background task that periodically health-checks all workers.

    Also runs periodic mDNS discovery to auto-register new workers.
    """

    DISCOVERY_INTERVAL = 30.0  # seconds between mDNS scans

    def __init__(
        self,
        cluster: ClusterState,
        client_factory: Callable[[str, str, int], object],
        config: HealthConfig,
        exclude_name: str | None = None,
        head_url: str | None = None,
    ) -> None:
        self._cluster = cluster
        self._client_factory = client_factory
        self._config = config
        self._exclude_name = exclude_name
        self._head_url = head_url
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._discovery_task: asyncio.Task | None = None  # type: ignore[type-arg]

    async def start(self) -> None:
        """Launch the background health check and discovery loops."""
        self._task = asyncio.create_task(self._check_loop())
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        logger.info(
            "Health checker started (interval=%.1fs, threshold=%d)",
            self._config.check_interval,
            self._config.failure_threshold,
        )

    async def stop(self) -> None:
        """Cancel the background tasks."""
        for task in (self._task, self._discovery_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("Health checker stopped")

    async def _check_loop(self) -> None:
        """Infinite loop: check each worker, sleep, repeat."""
        while True:
            try:
                await self._check_all_workers()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in health check loop")
            await asyncio.sleep(self._config.check_interval)

    async def _discovery_loop(self) -> None:
        """Periodically scan for new workers via mDNS."""
        while True:
            await asyncio.sleep(self.DISCOVERY_INTERVAL)
            try:
                count = await auto_discover_workers(
                    self._cluster,
                    timeout=2.0,
                    exclude_name=self._exclude_name,
                )
                if count:
                    logger.info("Auto-discovery found %d new worker(s)", count)
            except ImportError:
                pass  # zeroconf not installed
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Auto-discovery error", exc_info=True)

    async def _check_all_workers(self) -> None:
        """Check health of every registered worker concurrently."""
        workers = self._cluster.get_workers()
        tasks = []
        for worker in workers:
            if self._cluster.get_worker(worker.name) is None:
                continue
            # Skip the head node — it's always healthy by definition
            if worker.name == self._exclude_name:
                continue
            tasks.append(self._check_one_worker(worker.name, worker.host, worker.port))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_one_worker(self, name: str, host: str, port: int) -> None:
        """Check a single worker in a thread to avoid blocking the event loop."""
        from silo import __version__

        loop = asyncio.get_running_loop()
        try:
            client = self._client_factory(name, host, port)
            result = await loop.run_in_executor(
                None, client._get, "/health"  # type: ignore[union-attr]
            )
            worker_version = None
            if isinstance(result, dict):
                worker_version = result.get("version")
                if worker_version and worker_version != __version__:
                    logger.warning(
                        "Version mismatch: worker '%s' runs %s, head runs %s",
                        name,
                        worker_version,
                        __version__,
                    )
            self._cluster.record_health_success(name, version=worker_version)
            # Announce head URL to the worker using the IP that routes to it
            if self._head_url and name != self._exclude_name:
                try:
                    head_url = self._head_url_for(host, port)
                    await loop.run_in_executor(
                        None,
                        client._post,  # type: ignore[union-attr]
                        "/announce-head",
                        {"url": head_url},
                    )
                except Exception:
                    pass  # Non-critical
        except KeyError:
            pass  # Worker was unregistered during iteration
        except Exception:
            try:
                self._cluster.record_health_failure(name)
            except KeyError:
                pass  # Worker was unregistered

    def _head_url_for(self, worker_host: str, worker_port: int) -> str:
        """Get the head URL using the IP that routes to a specific worker.

        This handles VPNs (e.g., Netbird/WireGuard) where the head has
        different IPs on different interfaces.
        """
        import socket
        import urllib.parse

        parsed = urllib.parse.urlparse(self._head_url)
        head_port = parsed.port or 9900
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((worker_host, worker_port))
            local_ip = s.getsockname()[0]
            s.close()
            return f"http://{local_ip}:{head_port}"
        except Exception:
            return self._head_url  # type: ignore[return-value]


# ── Auto-discovery ───────────────────────────────


async def auto_discover_workers(
    cluster: ClusterState,
    timeout: float = 3.0,
    exclude_name: str | None = None,
) -> int:
    """Discover workers via mDNS and register them.

    Returns the number of workers registered.
    """
    import asyncio

    from silo.agent.discovery import discover_nodes

    # discover_nodes uses blocking time.sleep, so run in a thread
    loop = asyncio.get_running_loop()
    nodes = await loop.run_in_executor(None, discover_nodes, timeout)
    count = 0
    for node in nodes:
        if exclude_name and node.name == exclude_name:
            continue
        cluster.register_worker(node.name, node.host, node.port)
        count += 1
    logger.info("Auto-discovered %d worker(s)", count)
    return count
