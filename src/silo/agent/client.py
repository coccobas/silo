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
class NodeSystemStats:
    """CPU and GPU usage from any node."""

    cpu_percent: float
    gpu_percent: float
    gpu_name: str


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


def local_node_name() -> str:
    """Return the short hostname of this machine.

    This is the canonical identity for the local node — used instead of
    the hardcoded string "local" so that nodes are always identified by
    their actual machine name, whether local or remote.
    """
    import platform

    return platform.node().split(".")[0]


class LocalClient:
    """Client that calls local business logic directly."""

    NODE_NAME = local_node_name()

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

        result = spawn_model(
            name=name,
            repo_id=repo_id,
            host=host,
            port=port,
            quantize=quantize,
            output=output,
        )
        return result.pid

    def stop(self, name: str, grace_period: int = 30) -> bool:
        from silo.process.manager import stop_model

        return stop_model(name=name, grace_period=grace_period)

    def update(self, name: str, **kwargs) -> dict:
        """Update a running model server via the local daemon endpoint."""
        import json
        import urllib.request

        from silo.process.pid import read_pid_entry

        entry = read_pid_entry(name)
        if entry is None:
            return {"name": name, "restarted": False, "changes": []}

        server_url = f"http://{entry.host}:{entry.port}"

        def _post(path, data):
            body = json.dumps(data).encode()
            req = urllib.request.Request(
                f"{server_url}{path}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST" if "litellm" in path else "PUT",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())

        changes: list[str] = []

        if kwargs.get("litellm_enabled") is True and kwargs.get("litellm_url"):
            from silo.litellm.registry import normalize_litellm_url

            data: dict[str, str] = {"url": normalize_litellm_url(kwargs["litellm_url"])}
            if kwargs.get("litellm_api_key"):
                data["api_key"] = kwargs["litellm_api_key"]
            if kwargs.get("litellm_model_name"):
                data["model_name"] = kwargs["litellm_model_name"]
            _post("/admin/litellm/register", data)
            changes.append("litellm_registered")

        elif kwargs.get("litellm_enabled") is False:
            _post("/admin/litellm/deregister", {})
            changes.append("litellm_deregistered")

        if kwargs.get("model_name"):
            _post("/admin/model-name", {"model_name": kwargs["model_name"]})
            changes.append(f"model_name={kwargs['model_name']}")

        return {"name": name, "restarted": False, "changes": changes}

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

    def system_stats(self) -> NodeSystemStats:
        from silo.process.system_stats import get_system_stats

        stats = get_system_stats()
        return NodeSystemStats(
            cpu_percent=stats.cpu_percent,
            gpu_percent=stats.gpu_percent,
            gpu_name=stats.gpu_name,
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

    def update(self, name: str, **kwargs) -> dict:
        """Update a running model server on a remote node."""
        import json
        import urllib.request

        url = f"{self._base}/update"
        payload = {"name": name, **kwargs}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="PATCH",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def memory(self) -> NodeMemory:
        data = self._get("/memory")
        return NodeMemory(
            total_gb=data["total_gb"],
            available_gb=data["available_gb"],
            used_gb=data["used_gb"],
            pressure=data["pressure"],
            usage_percent=data["usage_percent"],
        )

    def system_stats(self) -> NodeSystemStats:
        data = self._get("/system-stats")
        return NodeSystemStats(
            cpu_percent=data["cpu_percent"],
            gpu_percent=data["gpu_percent"],
            gpu_name=data["gpu_name"],
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
    local = LocalClient()
    clients: dict[str, LocalClient | RemoteClient] = {
        local.NODE_NAME: local,
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


def resolve_head_url(app: Any = None) -> str | None:
    """Find the cluster head URL using multiple strategies.

    This is the canonical resolution function — used by the cluster screen,
    dashboard, and flow modal so they all find workers the same way.

    Strategies (tried in order):
        1. app.agent_head_port — TUI launched with --head
        2. app.cluster_head_url — explicit --head-url or mDNS discovered
        3. Local agent /head endpoint — head announced itself
        4. Probe config nodes for /cluster/status

    The result is cached on app.cluster_head_url when found via
    strategies 3 or 4, so subsequent calls are fast.
    """
    import json
    import urllib.request

    # 1. App-level head port (launched with --head)
    if app is not None:
        head_port = getattr(app, "agent_head_port", None)
        if head_port is not None:
            return f"http://127.0.0.1:{head_port}"

        # 2. App-level head URL (--head-url or mDNS discovered)
        cached = getattr(app, "cluster_head_url", None)
        if cached is not None:
            return cached

    # 3. Ask local agent if the head has announced itself
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:9900/head", timeout=1
        ) as resp:
            data = json.loads(resp.read())
            url = data.get("head_url")
            if url:
                if app is not None:
                    app.cluster_head_url = url
                return url
    except Exception:
        pass

    # 4. Probe config nodes for /cluster/status
    try:
        from silo.config.loader import load_config

        config = load_config()
        for node in config.nodes:
            probe_url = f"http://{node.host}:{node.port}"
            try:
                urllib.request.urlopen(
                    f"{probe_url}/cluster/status", timeout=2
                )
                if app is not None:
                    app.cluster_head_url = probe_url
                return probe_url
            except Exception:
                continue
    except Exception:
        pass

    return None


def fetch_cluster_workers(app: Any = None) -> list[str]:
    """Fetch worker node names from the cluster head.

    Uses resolve_head_url to find the head, then queries /cluster/status.
    Returns an empty list if no head is found or the request fails.
    """
    import json
    import urllib.request

    head_url = resolve_head_url(app)
    if head_url is None:
        return []

    try:
        req = urllib.request.Request(f"{head_url}/cluster/status")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return [w["name"] for w in data.get("workers", []) if "name" in w]
    except Exception:
        return []
