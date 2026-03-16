"""Node client — unified interface for local and remote management.

All methods return the same data structures regardless of whether the
node is local (direct function calls) or remote (HTTP to agent daemon).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from silo.agent.retry import RetryConfig, with_retry
from silo.config.models import NodeConfig


@dataclass(frozen=True)
class NodeProcess:
    """Process info from any node."""

    name: str
    pid: int
    port: int
    repo_id: str
    status: str
    node: str


@dataclass(frozen=True)
class NodeMemory:
    """Memory info from any node."""

    total_gb: float
    available_gb: float
    used_gb: float
    pressure: str
    usage_percent: float


@dataclass(frozen=True)
class NodeCheck:
    """Check result from any node."""

    name: str
    status: str
    message: str


@dataclass(frozen=True)
class NodeRegistryEntry:
    """Registry entry from any node."""

    repo_id: str
    format: str
    local_path: str | None = None
    size_bytes: int | None = None
    downloaded_at: str | None = None
    tags: list[str] = field(default_factory=list)


class LocalClient:
    """Client that calls local business logic directly."""

    NODE_NAME = "local"

    def list_processes(self) -> list[NodeProcess]:
        from silo.process.manager import list_running

        return [
            NodeProcess(
                name=p.name,
                pid=p.pid,
                port=p.port,
                repo_id=p.repo_id,
                status=p.status,
                node=self.NODE_NAME,
            )
            for p in list_running()
        ]

    def get_status(
        self, name: str, port: int = 0, repo_id: str = ""
    ) -> NodeProcess:
        from silo.process.manager import get_status

        info = get_status(name, port=port, repo_id=repo_id)
        return NodeProcess(
            name=info.name,
            pid=info.pid,
            port=info.port,
            repo_id=info.repo_id,
            status=info.status,
            node=self.NODE_NAME,
        )

    def spawn(
        self,
        name: str,
        repo_id: str,
        host: str = "127.0.0.1",
        port: int = 8800,
        quantize: str | None = None,
        output: str | None = None,
    ) -> int:
        from silo.process.manager import spawn_model

        return spawn_model(
            name=name,
            repo_id=repo_id,
            host=host,
            port=port,
            quantize=quantize,
            output=output,
        )

    def stop(self, name: str, grace_period: int = 30) -> bool:
        from silo.process.manager import stop_model

        return stop_model(name=name, grace_period=grace_period)

    def memory(self) -> NodeMemory:
        from silo.process.memory import get_memory_info

        mem = get_memory_info()
        return NodeMemory(
            total_gb=mem.total_gb,
            available_gb=mem.available_gb,
            used_gb=mem.used_gb,
            pressure=mem.pressure,
            usage_percent=mem.usage_percent,
        )

    def registry(self) -> list[NodeRegistryEntry]:
        from silo.registry.store import Registry

        return [
            NodeRegistryEntry(
                repo_id=e.repo_id,
                format=str(e.format),
                local_path=e.local_path,
                size_bytes=e.size_bytes,
                downloaded_at=e.downloaded_at,
                tags=list(e.tags),
            )
            for e in Registry.load().list()
        ]

    def doctor(self) -> list[NodeCheck]:
        from silo.doctor.checks import run_all_checks

        return [
            NodeCheck(name=c.name, status=c.status, message=c.message)
            for c in run_all_checks()
        ]

    def download(
        self, repo_id: str, local_dir: str | None = None
    ) -> str:
        from silo.download.hf import download_model, get_model_info
        from silo.registry.detector import detect_model_format
        from silo.registry.models import RegistryEntry
        from silo.registry.store import Registry

        path = download_model(repo_id, local_dir=local_dir)
        info = get_model_info(repo_id)
        fmt = detect_model_format(repo_id, info.get("siblings"))
        entry = RegistryEntry(
            repo_id=repo_id, format=fmt, local_path=str(path)
        )
        reg = Registry.load()
        updated = reg.add(entry)
        updated.save()
        return str(path)


class RemoteClient:
    """Client that talks to a remote Silo agent over HTTP."""

    def __init__(
        self,
        node: NodeConfig,
        retry_config: RetryConfig = RetryConfig(),
    ) -> None:
        self._node = node
        self._base = f"http://{node.host}:{node.port}"
        self._retry = retry_config

    @property
    def NODE_NAME(self) -> str:
        return self._node.name

    def _do_get(self, path: str) -> Any:
        import json
        import urllib.request

        url = f"{self._base}{path}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def _do_post(self, path: str, data: dict) -> Any:
        import json
        import urllib.request

        url = f"{self._base}{path}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> Any:
        return with_retry(self._do_get, self._retry, path)

    def _post(self, path: str, data: dict) -> Any:
        return with_retry(self._do_post, self._retry, path, data)

    def list_processes(self) -> list[NodeProcess]:
        data = self._get("/processes")
        return [
            NodeProcess(
                name=p["name"],
                pid=p["pid"],
                port=p["port"],
                repo_id=p["repo_id"],
                status=p["status"],
                node=self.NODE_NAME,
            )
            for p in data
        ]

    def get_status(
        self, name: str, port: int = 0, repo_id: str = ""
    ) -> NodeProcess:
        # Use the full status endpoint and find the matching process
        processes = self.list_processes()
        for p in processes:
            if p.name == name:
                return p
        return NodeProcess(
            name=name,
            pid=0,
            port=port,
            repo_id=repo_id,
            status="stopped",
            node=self.NODE_NAME,
        )

    def spawn(
        self,
        name: str,
        repo_id: str,
        host: str = "127.0.0.1",
        port: int = 8800,
        quantize: str | None = None,
        output: str | None = None,
    ) -> int:
        data = self._post(
            "/spawn",
            {
                "name": name,
                "repo_id": repo_id,
                "host": host,
                "port": port,
                "quantize": quantize,
                "output": output,
            },
        )
        return data["pid"]

    def stop(self, name: str, grace_period: int = 30) -> bool:
        data = self._post(
            "/stop", {"name": name, "grace_period": grace_period}
        )
        return data["stopped"]

    def memory(self) -> NodeMemory:
        data = self._get("/memory")
        return NodeMemory(
            total_gb=data["total_gb"],
            available_gb=data["available_gb"],
            used_gb=data["used_gb"],
            pressure=data["pressure"],
            usage_percent=data["usage_percent"],
        )

    def registry(self) -> list[NodeRegistryEntry]:
        data = self._get("/registry")
        return [
            NodeRegistryEntry(
                repo_id=e["repo_id"],
                format=e["format"],
                local_path=e.get("local_path"),
                size_bytes=e.get("size_bytes"),
                downloaded_at=e.get("downloaded_at"),
                tags=e.get("tags", []),
            )
            for e in data
        ]

    def doctor(self) -> list[NodeCheck]:
        data = self._get("/doctor")
        return [
            NodeCheck(name=c["name"], status=c["status"], message=c["message"])
            for c in data
        ]

    def download(
        self, repo_id: str, local_dir: str | None = None
    ) -> str:
        data = self._post(
            "/download",
            {"repo_id": repo_id, "local_dir": local_dir},
        )
        return data["local_path"]


def build_clients(
    nodes: list[NodeConfig] | None = None,
    retry_config: RetryConfig = RetryConfig(),
    discover: bool = False,
    discover_timeout: float = 3.0,
) -> dict[str, LocalClient | RemoteClient]:
    """Build a dict of node_name → client for all configured nodes.

    Always includes a 'local' client for the current machine.
    When *discover* is True, also scans the LAN via mDNS for agent nodes.
    Explicitly configured nodes take priority over discovered ones.
    """
    clients: dict[str, LocalClient | RemoteClient] = {
        "local": LocalClient(),
    }
    configured_names: set[str] = set()
    for node in nodes or []:
        clients[node.name] = RemoteClient(node, retry_config=retry_config)
        configured_names.add(node.name)

    if discover:
        from silo.agent.discovery import discover_nodes

        for found in discover_nodes(timeout=discover_timeout):
            if found.name not in configured_names and found.name not in clients:
                node_cfg = NodeConfig(
                    name=found.name, host=found.host, port=found.port
                )
                clients[found.name] = RemoteClient(
                    node_cfg, retry_config=retry_config
                )

    return clients
