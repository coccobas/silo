"""mDNS service discovery for Silo agent nodes.

Uses zeroconf (optional dependency) to advertise and discover
Silo agent daemons on the local network.

Install with: pip install silo[discovery]
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MDNS_SERVICE_TYPE = "_silo._tcp.local."


def is_discovery_available() -> bool:
    """Check whether zeroconf is installed."""
    try:
        import zeroconf  # noqa: F401

        return True
    except ImportError:
        return False


def _require_zeroconf() -> None:
    if not is_discovery_available():
        raise ImportError(
            "zeroconf is required for discovery. "
            "Install with: pip install silo[discovery]"
        )


# Re-export zeroconf classes lazily to avoid import errors
# when zeroconf is not installed.
def _import_zeroconf():  # noqa: ANN202
    from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf

    return Zeroconf, ServiceInfo, ServiceBrowser


# Make these available at module level for patching in tests.
# When zeroconf is installed, these are the real classes.
# Tests patch these names directly.
try:
    Zeroconf, ServiceInfo, ServiceBrowser = _import_zeroconf()
except ImportError:
    Zeroconf = None  # type: ignore[assignment,misc]
    ServiceInfo = None  # type: ignore[assignment,misc]
    ServiceBrowser = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class DiscoveredNode:
    """An agent node found via mDNS discovery."""

    name: str
    host: str
    port: int
    hostname: str
    role: str = "worker"


class ServiceAdvertiser:
    """Context manager that registers a Silo agent on the local network.

    Usage::

        with ServiceAdvertiser(node_name="mini-1", port=9900):
            # Service is advertised while inside the block
            run_server()
    """

    def __init__(
        self, node_name: str, port: int, role: str = "worker"
    ) -> None:
        _require_zeroconf()
        self._node_name = node_name
        self._port = port
        self._role = role
        self._zeroconf = None
        self._info = None

    def __enter__(self) -> ServiceAdvertiser:
        hostname = platform.node()
        self._info = ServiceInfo(
            type_=MDNS_SERVICE_TYPE,
            name=f"{self._node_name}.{MDNS_SERVICE_TYPE}",
            port=self._port,
            properties={
                b"node_name": self._node_name.encode(),
                b"hostname": hostname.encode(),
                b"role": self._role.encode(),
            },
            server=f"{hostname}.local.",
        )
        self._zeroconf = Zeroconf()
        self._zeroconf.register_service(self._info)
        logger.info(
            "Advertising %s on port %d via mDNS", self._node_name, self._port
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self._zeroconf and self._info:
            self._zeroconf.unregister_service(self._info)
            self._zeroconf.close()
            logger.info("Stopped advertising %s", self._node_name)


class _DiscoveryListener:
    """Collects service names as they are discovered."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._service_names: list[str] = []

    def add_service(self, zc, type_: str, name: str) -> None:  # noqa: ANN001
        with self._lock:
            self._service_names.append(name)

    def remove_service(self, zc, type_: str, name: str) -> None:  # noqa: ANN001
        pass  # Not needed for one-shot discovery

    def update_service(self, zc, type_: str, name: str) -> None:  # noqa: ANN001
        pass  # Not needed for one-shot discovery

    @property
    def names(self) -> list[str]:
        with self._lock:
            return list(self._service_names)


def discover_nodes(timeout: float = 3.0) -> list[DiscoveredNode]:
    """Scan the local network for Silo agent daemons.

    Args:
        timeout: How long to listen for mDNS responses (seconds).

    Returns:
        List of discovered nodes. Empty if none found or zeroconf
        is not installed.

    Raises:
        ImportError: If zeroconf is not installed.
    """
    _require_zeroconf()

    zc = Zeroconf()
    listener = _DiscoveryListener()
    try:
        ServiceBrowser(zc, MDNS_SERVICE_TYPE, listener)
        time.sleep(timeout)

        nodes: list[DiscoveredNode] = []
        for service_name in listener.names:
            info = zc.get_service_info(MDNS_SERVICE_TYPE, service_name)
            if info is None:
                logger.debug("Could not resolve %s, skipping", service_name)
                continue

            props = info.properties or {}
            node_name = props.get(b"node_name", b"unknown").decode()
            hostname = props.get(b"hostname", b"unknown").decode()
            role = props.get(b"role", b"worker").decode()
            addresses = info.parsed_addresses()
            host = addresses[0] if addresses else "unknown"

            nodes.append(
                DiscoveredNode(
                    name=node_name,
                    host=host,
                    port=info.port,
                    hostname=hostname,
                    role=role,
                )
            )

        logger.info("Discovered %d node(s) on the network", len(nodes))
        return nodes
    finally:
        zc.close()
