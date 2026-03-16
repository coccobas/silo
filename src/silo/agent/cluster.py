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
    """

    def __init__(self, config: HealthConfig) -> None:
        self._config = config
        self._workers: dict[str, WorkerNode] = {}
        self._lock = asyncio.Lock()

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
        logger.info("Registered worker '%s' at %s:%d", name, host, port)
        return worker

    def unregister_worker(self, name: str) -> bool:
        """Remove a worker. Returns True if it existed."""
        if name in self._workers:
            del self._workers[name]
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

    def record_health_success(self, name: str) -> WorkerNode:
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
    """Background task that periodically health-checks all workers."""

    def __init__(
        self,
        cluster: ClusterState,
        client_factory: Callable[[str, str, int], object],
        config: HealthConfig,
    ) -> None:
        self._cluster = cluster
        self._client_factory = client_factory
        self._config = config
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]

    async def start(self) -> None:
        """Launch the background health check loop."""
        self._task = asyncio.create_task(self._check_loop())
        logger.info(
            "Health checker started (interval=%.1fs, threshold=%d)",
            self._config.check_interval,
            self._config.failure_threshold,
        )

    async def stop(self) -> None:
        """Cancel the background task."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
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

    async def _check_all_workers(self) -> None:
        """Check health of every registered worker."""
        workers = self._cluster.get_workers()
        for worker in workers:
            # Re-check the worker still exists (could have been unregistered)
            if self._cluster.get_worker(worker.name) is None:
                continue
            try:
                client = self._client_factory(
                    worker.name, worker.host, worker.port
                )
                client._get("/health")  # type: ignore[union-attr]
                self._cluster.record_health_success(worker.name)
            except KeyError:
                pass  # Worker was unregistered during iteration
            except Exception:
                try:
                    self._cluster.record_health_failure(worker.name)
                except KeyError:
                    pass  # Worker was unregistered


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
