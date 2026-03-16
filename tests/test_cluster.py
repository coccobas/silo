"""Tests for cluster state management, placement, and health checking."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from silo.agent.cluster import ClusterState, HealthChecker, auto_discover_workers, select_node
from silo.agent.cluster_schemas import HealthConfig, WorkerNode


# ── ClusterState: registration ───────────────────


class TestClusterStateRegistration:
    def test_register_new_worker(self):
        cluster = ClusterState(HealthConfig())
        worker = cluster.register_worker("mini-1", "10.0.0.5", 9900)
        assert worker.name == "mini-1"
        assert worker.host == "10.0.0.5"
        assert worker.port == 9900
        assert worker.status == "unknown"
        assert worker.consecutive_failures == 0

    def test_register_returns_frozen(self):
        cluster = ClusterState(HealthConfig())
        worker = cluster.register_worker("x", "1.2.3.4", 9900)
        with pytest.raises(Exception):
            worker.name = "y"  # type: ignore[misc]

    def test_register_existing_updates_host_port(self):
        cluster = ClusterState(HealthConfig())
        cluster.register_worker("mini-1", "10.0.0.5", 9900)
        updated = cluster.register_worker("mini-1", "10.0.0.99", 8000)
        assert updated.host == "10.0.0.99"
        assert updated.port == 8000

    def test_register_preserves_health_state(self):
        cluster = ClusterState(HealthConfig())
        cluster.register_worker("mini-1", "10.0.0.5", 9900)
        cluster.record_health_success("mini-1")
        updated = cluster.register_worker("mini-1", "10.0.0.5", 9900)
        assert updated.status == "healthy"

    def test_unregister_worker(self):
        cluster = ClusterState(HealthConfig())
        cluster.register_worker("mini-1", "10.0.0.5", 9900)
        assert cluster.unregister_worker("mini-1") is True
        assert cluster.get_worker("mini-1") is None

    def test_unregister_nonexistent(self):
        cluster = ClusterState(HealthConfig())
        assert cluster.unregister_worker("ghost") is False

    def test_get_worker(self):
        cluster = ClusterState(HealthConfig())
        cluster.register_worker("mini-1", "10.0.0.5", 9900)
        worker = cluster.get_worker("mini-1")
        assert worker is not None
        assert worker.name == "mini-1"

    def test_get_worker_nonexistent(self):
        cluster = ClusterState(HealthConfig())
        assert cluster.get_worker("nope") is None

    def test_get_workers_returns_tuple(self):
        cluster = ClusterState(HealthConfig())
        cluster.register_worker("a", "1.1.1.1", 9900)
        cluster.register_worker("b", "2.2.2.2", 9900)
        workers = cluster.get_workers()
        assert isinstance(workers, tuple)
        assert len(workers) == 2

    def test_get_healthy_workers(self):
        cluster = ClusterState(HealthConfig())
        cluster.register_worker("a", "1.1.1.1", 9900)
        cluster.register_worker("b", "2.2.2.2", 9900)
        cluster.record_health_success("a")
        healthy = cluster.get_healthy_workers()
        assert len(healthy) == 1
        assert healthy[0].name == "a"


# ── ClusterState: health tracking ────────────────


class TestClusterStateHealth:
    def test_record_success_marks_healthy(self):
        cluster = ClusterState(HealthConfig())
        cluster.register_worker("mini-1", "10.0.0.5", 9900)
        worker = cluster.record_health_success("mini-1")
        assert worker.status == "healthy"
        assert worker.consecutive_failures == 0

    def test_record_failure_increments(self):
        cluster = ClusterState(HealthConfig(failure_threshold=3))
        cluster.register_worker("mini-1", "10.0.0.5", 9900)
        cluster.record_health_success("mini-1")
        worker = cluster.record_health_failure("mini-1")
        assert worker.consecutive_failures == 1
        assert worker.status == "healthy"  # Not yet at threshold

    def test_record_failure_marks_unhealthy_at_threshold(self):
        cluster = ClusterState(HealthConfig(failure_threshold=3))
        cluster.register_worker("mini-1", "10.0.0.5", 9900)
        cluster.record_health_success("mini-1")
        cluster.record_health_failure("mini-1")
        cluster.record_health_failure("mini-1")
        worker = cluster.record_health_failure("mini-1")
        assert worker.consecutive_failures == 3
        assert worker.status == "unhealthy"

    def test_record_success_resets_failures(self):
        cluster = ClusterState(HealthConfig(failure_threshold=3))
        cluster.register_worker("mini-1", "10.0.0.5", 9900)
        cluster.record_health_failure("mini-1")
        cluster.record_health_failure("mini-1")
        worker = cluster.record_health_success("mini-1")
        assert worker.consecutive_failures == 0
        assert worker.status == "healthy"

    def test_record_health_nonexistent_raises(self):
        cluster = ClusterState(HealthConfig())
        with pytest.raises(KeyError):
            cluster.record_health_success("ghost")
        with pytest.raises(KeyError):
            cluster.record_health_failure("ghost")


# ── Placement: select_node ───────────────────────


class TestSelectNode:
    @pytest.mark.asyncio
    async def test_preferred_node_valid(self):
        cluster = ClusterState(HealthConfig())
        cluster.register_worker("mini-1", "10.0.0.5", 9900)
        cluster.record_health_success("mini-1")

        mock_client = MagicMock()
        clients = {"mini-1": mock_client}

        name, client = await select_node(cluster, clients, preferred_node="mini-1")
        assert name == "mini-1"
        assert client is mock_client

    @pytest.mark.asyncio
    async def test_preferred_node_unhealthy_raises(self):
        cluster = ClusterState(HealthConfig(failure_threshold=1))
        cluster.register_worker("mini-1", "10.0.0.5", 9900)
        cluster.record_health_failure("mini-1")

        with pytest.raises(ValueError, match="unhealthy"):
            await select_node(cluster, {}, preferred_node="mini-1")

    @pytest.mark.asyncio
    async def test_preferred_node_nonexistent_raises(self):
        cluster = ClusterState(HealthConfig())
        with pytest.raises(ValueError, match="not found"):
            await select_node(cluster, {}, preferred_node="ghost")

    @pytest.mark.asyncio
    async def test_picks_most_available_memory(self):
        cluster = ClusterState(HealthConfig())
        cluster.register_worker("a", "1.1.1.1", 9900)
        cluster.register_worker("b", "2.2.2.2", 9900)
        cluster.record_health_success("a")
        cluster.record_health_success("b")

        client_a = MagicMock()
        client_a.memory.return_value = MagicMock(available_gb=10.0)
        client_b = MagicMock()
        client_b.memory.return_value = MagicMock(available_gb=40.0)
        clients = {"a": client_a, "b": client_b}

        name, client = await select_node(cluster, clients)
        assert name == "b"
        assert client is client_b

    @pytest.mark.asyncio
    async def test_skips_unreachable_node(self):
        cluster = ClusterState(HealthConfig())
        cluster.register_worker("a", "1.1.1.1", 9900)
        cluster.register_worker("b", "2.2.2.2", 9900)
        cluster.record_health_success("a")
        cluster.record_health_success("b")

        client_a = MagicMock()
        client_a.memory.side_effect = ConnectionError("down")
        client_b = MagicMock()
        client_b.memory.return_value = MagicMock(available_gb=20.0)
        clients = {"a": client_a, "b": client_b}

        name, client = await select_node(cluster, clients)
        assert name == "b"

    @pytest.mark.asyncio
    async def test_no_healthy_workers_raises(self):
        cluster = ClusterState(HealthConfig())
        with pytest.raises(RuntimeError, match="No healthy workers"):
            await select_node(cluster, {})

    @pytest.mark.asyncio
    async def test_all_unreachable_raises(self):
        cluster = ClusterState(HealthConfig())
        cluster.register_worker("a", "1.1.1.1", 9900)
        cluster.record_health_success("a")

        client_a = MagicMock()
        client_a.memory.side_effect = ConnectionError("down")
        clients = {"a": client_a}

        with pytest.raises(RuntimeError, match="No reachable workers"):
            await select_node(cluster, clients)


# ── HealthChecker ────────────────────────────────


class TestHealthChecker:
    @pytest.mark.asyncio
    async def test_marks_healthy_on_success(self):
        cluster = ClusterState(HealthConfig(check_interval=0.05))
        cluster.register_worker("mini-1", "10.0.0.5", 9900)

        mock_client = MagicMock()
        mock_client._get.return_value = {"status": "ok"}

        def factory(name, host, port):
            return mock_client

        checker = HealthChecker(cluster, factory, cluster._config)
        await checker.start()
        await asyncio.sleep(0.15)
        await checker.stop()

        worker = cluster.get_worker("mini-1")
        assert worker is not None
        assert worker.status == "healthy"

    @pytest.mark.asyncio
    async def test_marks_unhealthy_after_failures(self):
        config = HealthConfig(check_interval=0.05, failure_threshold=2)
        cluster = ClusterState(config)
        cluster.register_worker("mini-1", "10.0.0.5", 9900)

        mock_client = MagicMock()
        mock_client._get.side_effect = ConnectionError("down")

        def factory(name, host, port):
            return mock_client

        checker = HealthChecker(cluster, factory, config)
        await checker.start()
        await asyncio.sleep(0.2)
        await checker.stop()

        worker = cluster.get_worker("mini-1")
        assert worker is not None
        assert worker.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        cluster = ClusterState(HealthConfig(check_interval=60))
        checker = HealthChecker(cluster, lambda n, h, p: MagicMock(), cluster._config)
        await checker.start()
        assert checker._task is not None
        await checker.stop()
        assert checker._task.cancelled() or checker._task.done()


# ── Auto-discovery ───────────────────────────────


class TestAutoDiscoverWorkers:
    @pytest.mark.asyncio
    @patch("silo.agent.discovery.discover_nodes")
    async def test_registers_found_nodes(self, mock_discover):
        from silo.agent.discovery import DiscoveredNode

        mock_discover.return_value = [
            DiscoveredNode(name="mini-1", host="10.0.0.5", port=9900, hostname="mini-1.local"),
            DiscoveredNode(name="studio", host="10.0.0.10", port=8000, hostname="studio.local"),
        ]
        cluster = ClusterState(HealthConfig())
        count = await auto_discover_workers(cluster, timeout=1.0)
        assert count == 2
        assert cluster.get_worker("mini-1") is not None
        assert cluster.get_worker("studio") is not None

    @pytest.mark.asyncio
    @patch("silo.agent.discovery.discover_nodes")
    async def test_empty_network(self, mock_discover):
        mock_discover.return_value = []
        cluster = ClusterState(HealthConfig())
        count = await auto_discover_workers(cluster)
        assert count == 0

    @pytest.mark.asyncio
    @patch("silo.agent.discovery.discover_nodes")
    async def test_skips_self(self, mock_discover):
        from silo.agent.discovery import DiscoveredNode

        mock_discover.return_value = [
            DiscoveredNode(name="head", host="10.0.0.1", port=9900, hostname="head.local"),
            DiscoveredNode(name="worker", host="10.0.0.2", port=9900, hostname="worker.local"),
        ]
        cluster = ClusterState(HealthConfig())
        count = await auto_discover_workers(cluster, timeout=1.0, exclude_name="head")
        assert count == 1
        assert cluster.get_worker("head") is None
        assert cluster.get_worker("worker") is not None
