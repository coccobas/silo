"""TUI screens."""

from silo.tui.screens.cluster import ClusterScreen
from silo.tui.screens.dashboard import DashboardScreen
from silo.tui.screens.doctor import DoctorScreen
from silo.tui.screens.flows import FlowsScreen
from silo.tui.screens.models import ModelsScreen
from silo.tui.screens.servers import ServersScreen

__all__ = [
    "ClusterScreen",
    "DashboardScreen",
    "DoctorScreen",
    "FlowsScreen",
    "ModelsScreen",
    "ServersScreen",
]
