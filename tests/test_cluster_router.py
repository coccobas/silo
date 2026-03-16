"""Integration tests for cluster router endpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from silo.agent.cluster import ClusterState
from silo.agent.cluster_schemas import HealthConfig


@pytest.fixture
def cluster():
    return ClusterState(HealthConfig())


@pytest.fixture
def _mock_deps(monkeypatch):
    """Mock business logic for daemon tests."""

    @dataclass(frozen=True)
    class FakeProc:
        name: str = "test-model"
        pid: int = 1234
        port: int = 8800
        repo_id: str = "org/model"
        status: str = "running"

    @dataclass(frozen=True)
    class FakeMem:
        total_gb: float = 64.0
        available_gb: float = 40.0
        used_gb: float = 24.0
        pressure: str = "normal"

        @property
        def usage_percent(self):
            return (self.used_gb / self.total_gb) * 100

    @dataclass(frozen=True)
    class FakeCheck:
        name: str = "python"
        status: str = "ok"
        message: str = "3.12"

    monkeypatch.setattr(
        "silo.process.manager.list_running", lambda **kw: [FakeProc()]
    )
    monkeypatch.setattr(
        "silo.process.manager.spawn_model", lambda **kw: 5678
    )
    monkeypatch.setattr(
        "silo.process.manager.stop_model", lambda **kw: True
    )
    monkeypatch.setattr(
        "silo.process.memory.get_memory_info", lambda: FakeMem()
    )
    monkeypatch.setattr(
        "silo.doctor.checks.run_all_checks", lambda: [FakeCheck()]
    )

    from silo.registry.models import ModelFormat, RegistryEntry
    from silo.registry.store import Registry

    entry = RegistryEntry(repo_id="org/model", format=ModelFormat.MLX)
    monkeypatch.setattr(
        "silo.registry.store.Registry.load",
        classmethod(lambda cls, **kw: Registry({entry.repo_id: entry})),
    )


@pytest.fixture
def _mock_network():
    """Mock all network-dependent discovery/advertising."""
    mock_advertiser = MagicMock()
    mock_advertiser.__enter__ = MagicMock(return_value=mock_advertiser)
    mock_advertiser.__exit__ = MagicMock(return_value=False)
    with (
        patch(
            "silo.agent.discovery.ServiceAdvertiser",
            return_value=mock_advertiser,
        ),
        patch("silo.agent.discovery.discover_nodes", return_value=[]),
    ):
        yield


@pytest.fixture
def head_client(_mock_deps, _mock_network):
    """TestClient for a head-mode daemon."""
    from silo.agent.daemon import create_agent_app

    app = create_agent_app(node_name="head", port=9900, head=True)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def normal_client(_mock_deps, _mock_network):
    """TestClient for a non-head daemon."""
    from silo.agent.daemon import create_agent_app

    app = create_agent_app(node_name="worker", port=9900, head=False)
    with TestClient(app) as client:
        yield client


# ── Head vs normal mode ──────────────────────────


class TestClusterEndpointAvailability:
    def test_head_has_cluster_status(self, head_client):
        resp = head_client.get("/cluster/status")
        assert resp.status_code == 200

    def test_normal_has_no_cluster_status(self, normal_client):
        resp = normal_client.get("/cluster/status")
        assert resp.status_code == 404

    def test_head_has_health(self, head_client):
        """Head is still a normal agent too."""
        resp = head_client.get("/health")
        assert resp.status_code == 200

    def test_head_has_processes(self, head_client):
        resp = head_client.get("/processes")
        assert resp.status_code == 200


# ── Worker registration ──────────────────────────


class TestClusterRegister:
    def test_register_worker(self, head_client):
        resp = head_client.post(
            "/cluster/register",
            json={"name": "mini-1", "host": "10.0.0.5", "port": 9900},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "mini-1"
        assert data["status"] == "unknown"

    def test_register_duplicate_updates(self, head_client):
        head_client.post(
            "/cluster/register",
            json={"name": "mini-1", "host": "10.0.0.5", "port": 9900},
        )
        resp = head_client.post(
            "/cluster/register",
            json={"name": "mini-1", "host": "10.0.0.99", "port": 8000},
        )
        assert resp.status_code == 200
        assert resp.json()["host"] == "10.0.0.99"
        assert resp.json()["port"] == 8000


# ── Cluster status ───────────────────────────────


class TestClusterStatus:
    def test_empty_cluster(self, head_client):
        resp = head_client.get("/cluster/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["head"] == "head"
        # Head registers itself as a worker
        assert len(data["workers"]) >= 1

    def test_with_registered_workers(self, head_client):
        head_client.post(
            "/cluster/register",
            json={"name": "mini-1", "host": "10.0.0.5", "port": 9900},
        )
        resp = head_client.get("/cluster/status")
        data = resp.json()
        names = [w["name"] for w in data["workers"]]
        assert "mini-1" in names


# ── Cluster spawn ────────────────────────────────


class TestClusterSpawn:
    def test_spawn_on_self(self, head_client):
        """Head node spawns locally when it's the only healthy worker."""
        resp = head_client.post(
            "/cluster/spawn",
            json={
                "name": "my-model",
                "repo_id": "org/model",
                "node": "head",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node"] == "head"
        assert data["pid"] == 5678

    def test_spawn_preferred_nonexistent(self, head_client):
        resp = head_client.post(
            "/cluster/spawn",
            json={
                "name": "my-model",
                "repo_id": "org/model",
                "node": "ghost",
            },
        )
        assert resp.status_code == 400


# ── Cluster stop ─────────────────────────────────


class TestClusterStop:
    def test_stop_on_self(self, head_client):
        """Head can stop a model running locally."""
        resp = head_client.post(
            "/cluster/stop",
            json={"name": "test-model"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stopped"] is True
        assert data["node"] == "head"

    def test_stop_not_found(self, head_client):
        resp = head_client.post(
            "/cluster/stop",
            json={"name": "nonexistent-model"},
        )
        assert resp.status_code == 404
