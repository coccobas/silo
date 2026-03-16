"""Silo agent — remote management daemon and discovery."""

from silo.agent.daemon import create_agent_app
from silo.agent.retry import RetryConfig

__all__ = ["RetryConfig", "create_agent_app"]

# Conditionally export discovery components
try:
    from silo.agent.discovery import (
        DiscoveredNode,
        ServiceAdvertiser,
        discover_nodes,
    )

    __all__ += ["DiscoveredNode", "ServiceAdvertiser", "discover_nodes"]
except ImportError:
    pass
