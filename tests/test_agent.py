"""Tests for the agent daemon, client abstraction, and CLI command."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Config model tests ───────────────────────────────


class TestNodeConfig:
    def test_defaults(self):
        from silo.config.models import NodeConfig

        node = NodeConfig(name="mini-1", host="192.168.1.50")
        assert node.name == "mini-1"
        assert node.host == "192.168.1.50"
        assert node.port == 9900

    def test_custom_port(self):
        from silo.config.models import NodeConfig

        node = NodeConfig(name="studio", host="10.0.0.5", port=8000)
        assert node.port == 8000

    def test_frozen(self):
        from silo.config.models import NodeConfig

        node = NodeConfig(name="x", host="1.2.3.4")
        with pytest.raises(Exception):
            node.name = "y"


class TestModelConfigNode:
    def test_node_default_none(self):
        from silo.config.models import ModelConfig

        m = ModelConfig(name="test", repo="org/model")
        assert m.node is None

    def test_node_assigned(self):
        from silo.config.models import ModelConfig

        m = ModelConfig(name="test", repo="org/model", node="mini-1")
        assert m.node == "mini-1"


class TestAppConfigNodes:
    def test_empty_nodes(self):
        from silo.config.models import AppConfig

        config = AppConfig()
        assert config.nodes == []

    def test_with_nodes(self):
        from silo.config.models import AppConfig, NodeConfig

        config = AppConfig(
            nodes=[NodeConfig(name="n1", host="10.0.0.1")]
        )
        assert len(config.nodes) == 1
        assert config.nodes[0].name == "n1"


# ── Agent daemon tests ───────────────────────────────


@pytest.fixture
def _mock_agent_deps(monkeypatch):
    """Mock business logic for agent daemon tests."""

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
        "silo.process.manager.list_running",
        lambda **kw: [FakeProc()],
    )
    monkeypatch.setattr(
        "silo.process.manager.spawn_model",
        lambda **kw: 5678,
    )
    monkeypatch.setattr(
        "silo.process.manager.stop_model",
        lambda **kw: True,
    )
    monkeypatch.setattr(
        "silo.process.memory.get_memory_info",
        lambda: FakeMem(),
    )
    monkeypatch.setattr(
        "silo.doctor.checks.run_all_checks",
        lambda: [FakeCheck()],
    )

    from silo.registry.models import ModelFormat, RegistryEntry
    from silo.registry.store import Registry

    entry = RegistryEntry(repo_id="org/model", format=ModelFormat.MLX)
    monkeypatch.setattr(
        "silo.registry.store.Registry.load",
        classmethod(lambda cls, **kw: Registry({entry.repo_id: entry})),
    )


@pytest.mark.usefixtures("_mock_agent_deps")
class TestAgentDaemon:
    @pytest.fixture
    def client(self):
        from silo.agent.daemon import create_agent_app
        return TestClient(create_agent_app())

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "hostname" in data

    def test_status(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "hostname" in data
        assert len(data["processes"]) == 1
        assert data["processes"][0]["name"] == "test-model"
        assert data["memory"]["pressure"] == "normal"
        assert len(data["registry"]) == 1

    def test_processes(self, client):
        resp = client.get("/processes")
        assert resp.status_code == 200
        procs = resp.json()
        assert len(procs) == 1
        assert procs[0]["status"] == "running"

    def test_spawn(self, client):
        resp = client.post(
            "/spawn",
            json={"name": "m1", "repo_id": "org/model"},
        )
        assert resp.status_code == 200
        assert resp.json()["pid"] == 5678

    def test_stop(self, client):
        resp = client.post("/stop", json={"name": "m1"})
        assert resp.status_code == 200
        assert resp.json()["stopped"] is True

    def test_memory(self, client):
        resp = client.get("/memory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_gb"] == 64.0
        assert data["pressure"] == "normal"

    def test_registry(self, client):
        resp = client.get("/registry")
        assert resp.status_code == 200
        entries = resp.json()
        assert len(entries) == 1
        assert entries[0]["repo_id"] == "org/model"

    def test_doctor(self, client):
        resp = client.get("/doctor")
        assert resp.status_code == 200
        checks = resp.json()
        assert len(checks) == 1
        assert checks[0]["status"] == "ok"


# ── Client abstraction tests ─────────────────────────


@pytest.mark.usefixtures("_mock_agent_deps")
class TestLocalClient:
    def test_list_processes(self):
        from silo.agent.client import LocalClient, local_node_name

        client = LocalClient()
        procs = client.list_processes()
        assert len(procs) == 1
        assert procs[0].node == local_node_name()
        assert procs[0].status == "running"

    def test_memory(self):
        from silo.agent.client import LocalClient

        client = LocalClient()
        mem = client.memory()
        assert mem.total_gb == 64.0
        assert mem.pressure == "normal"

    def test_doctor(self):
        from silo.agent.client import LocalClient

        client = LocalClient()
        checks = client.doctor()
        assert len(checks) == 1
        assert checks[0].status == "ok"

    def test_registry(self):
        from silo.agent.client import LocalClient

        client = LocalClient()
        entries = client.registry()
        assert len(entries) == 1
        assert entries[0].repo_id == "org/model"

    def test_spawn(self):
        from silo.agent.client import LocalClient

        client = LocalClient()
        pid = client.spawn(name="m1", repo_id="org/model")
        assert pid == 5678

    def test_stop(self):
        from silo.agent.client import LocalClient

        client = LocalClient()
        assert client.stop("m1") is True


class TestBuildClients:
    def test_local_only(self):
        from silo.agent.client import LocalClient, build_clients, local_node_name

        clients = build_clients()
        local_name = local_node_name()
        assert local_name in clients
        assert isinstance(clients[local_name], LocalClient)

    def test_with_remote_nodes(self):
        from silo.agent.client import RemoteClient, build_clients, local_node_name
        from silo.config.models import NodeConfig

        nodes = [
            NodeConfig(name="mini-1", host="10.0.0.1"),
            NodeConfig(name="studio", host="10.0.0.2", port=8000),
        ]
        clients = build_clients(nodes)
        assert local_node_name() in clients
        assert "mini-1" in clients
        assert "studio" in clients
        assert isinstance(clients["mini-1"], RemoteClient)
        assert clients["studio"]._base == "http://10.0.0.2:8000"


class TestRemoteClient:
    def test_node_name(self):
        from silo.agent.client import RemoteClient
        from silo.config.models import NodeConfig

        client = RemoteClient(NodeConfig(name="mini-1", host="10.0.0.1"))
        assert client.NODE_NAME == "mini-1"

    def test_base_url(self):
        from silo.agent.client import RemoteClient
        from silo.config.models import NodeConfig

        client = RemoteClient(
            NodeConfig(name="x", host="192.168.1.50", port=9900)
        )
        assert client._base == "http://192.168.1.50:9900"

    def test_accepts_retry_config(self):
        from silo.agent.client import RemoteClient
        from silo.agent.retry import RetryConfig
        from silo.config.models import NodeConfig

        config = RetryConfig(max_retries=5, base_delay=2.0)
        client = RemoteClient(
            NodeConfig(name="x", host="1.2.3.4"), retry_config=config
        )
        assert client._retry.max_retries == 5

    @patch("silo.agent.retry.time.sleep")
    def test_get_retries_on_connection_error(self, mock_sleep):
        import json
        from io import BytesIO
        from silo.agent.client import RemoteClient
        from silo.agent.retry import RetryConfig
        from silo.config.models import NodeConfig

        client = RemoteClient(
            NodeConfig(name="n", host="10.0.0.1"),
            retry_config=RetryConfig(max_retries=3),
        )
        response = MagicMock()
        response.read.return_value = json.dumps([]).encode()
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = [
                ConnectionError("refused"),
                ConnectionError("refused"),
                response,
            ]
            result = client._get("/processes")
            assert result == []
            assert mock_open.call_count == 3

    @patch("silo.agent.retry.time.sleep")
    def test_get_no_retry_on_404(self, mock_sleep):
        import urllib.error
        from silo.agent.client import RemoteClient
        from silo.agent.retry import RetryConfig
        from silo.config.models import NodeConfig

        client = RemoteClient(
            NodeConfig(name="n", host="10.0.0.1"),
            retry_config=RetryConfig(max_retries=3),
        )
        err = urllib.error.HTTPError(
            url="http://x", code=404, msg="Not Found", hdrs=None, fp=None
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(urllib.error.HTTPError):
                client._get("/missing")
        mock_sleep.assert_not_called()


# ── CLI command tests ─────────────────────────────────


class TestAgentCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["agent", "--help"])
        assert result.exit_code == 0
        assert "agent" in result.output.lower()

    def test_start_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["agent", "start", "--help"])
        assert result.exit_code == 0
        assert "bind" in result.output.lower() or "host" in result.output.lower()

    def test_discover_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["agent", "discover", "--help"])
        assert result.exit_code == 0
        assert "timeout" in result.output.lower()

    @patch("silo.agent.discovery.discover_nodes")
    def test_discover_shows_nodes(self, mock_discover, cli_runner, cli_app):
        from silo.agent.discovery import DiscoveredNode

        mock_discover.return_value = [
            DiscoveredNode(
                name="mini-1", host="10.0.0.5", port=9900, hostname="mini-1.local"
            ),
        ]
        result = cli_runner.invoke(cli_app, ["agent", "discover", "--timeout", "0.1"])
        assert result.exit_code == 0
        assert "mini-1" in result.output
        assert "10.0.0.5" in result.output

    @patch("silo.agent.discovery.discover_nodes")
    def test_discover_empty(self, mock_discover, cli_runner, cli_app):
        mock_discover.return_value = []
        result = cli_runner.invoke(cli_app, ["agent", "discover", "--timeout", "0.1"])
        assert result.exit_code == 0
        assert "no agents found" in result.output.lower()
