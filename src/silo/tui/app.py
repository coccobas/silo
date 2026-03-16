"""Textual TUI application for Silo — multi-screen management interface."""

from __future__ import annotations


def create_tui_app():
    """Create and return the Textual TUI application.

    Returns:
        A Textual App instance.

    Raises:
        ImportError: If textual is not installed.
    """
    try:
        from textual.app import App
        from textual.binding import Binding
    except ImportError as e:
        raise ImportError(
            "Textual is required for the TUI. "
            "Install with: uv pip install 'silo[tui]'"
        ) from e

    from silo.tui.screens.cluster import ClusterScreen
    from silo.tui.screens.dashboard import DashboardScreen
    from silo.tui.screens.doctor import DoctorScreen
    from silo.tui.screens.flows import FlowsScreen
    from silo.tui.screens.models import ModelsScreen
    from silo.tui.screens.servers import ServersScreen
    from silo.tui.widgets.download_tracker import DownloadTracker

    class HeLLMperApp(App):
        """TUI dashboard for managing Silo model servers."""

        TITLE = "Silo"
        CSS_PATH = "styles.tcss"
        downloads: DownloadTracker

        MODES = {
            "dashboard": DashboardScreen,
            "servers": ServersScreen,
            "models": ModelsScreen,
            "flows": FlowsScreen,
            "cluster": ClusterScreen,
            "doctor": DoctorScreen,
        }

        BINDINGS = [
            Binding("1", "switch_mode('dashboard')", "Dashboard", show=False),
            Binding("2", "switch_mode('servers')", "Servers", show=False),
            Binding("3", "switch_mode('models')", "Models", show=False),
            Binding("4", "switch_mode('flows')", "Flows", show=False),
            Binding("5", "switch_mode('cluster')", "Cluster", show=False),
            Binding("6", "switch_mode('doctor')", "Doctor", show=False),
            Binding("q", "request_quit", "Quit"),
        ]

        def on_mount(self) -> None:
            self.downloads = DownloadTracker()
            self.switch_mode("dashboard")


        def action_request_quit(self) -> None:
            from silo.tui.widgets.confirm_modal import ConfirmModal

            def on_confirm(confirmed: bool) -> None:
                if confirmed:
                    self.exit()

            self.push_screen(ConfirmModal("Quit Silo?"), on_confirm)

    return HeLLMperApp()
