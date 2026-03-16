"""Tests for TUI module and CLI command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestUiCommand:
    def test_help(self, cli_runner, cli_app):
        result = cli_runner.invoke(cli_app, ["ui", "--help"])
        assert result.exit_code == 0
        assert "TUI" in result.output or "dashboard" in result.output

    def test_ui_import_error(self, cli_runner, cli_app):
        """Test that the CLI gracefully handles missing textual."""
        with patch.dict("sys.modules", {"silo.tui.app": None}):
            result = cli_runner.invoke(cli_app, ["ui"])
        assert result.exit_code == 1
        assert "Textual" in result.output or "tui" in result.output.lower()


class TestCreateTuiApp:
    def test_creates_app(self):
        from silo.tui.app import create_tui_app

        app = create_tui_app()
        assert app.__class__.__name__ == "HeLLMperApp"

    def test_modes_registered(self):
        from silo.tui.app import create_tui_app

        app = create_tui_app()
        expected = {"dashboard", "servers", "models", "flows", "cluster", "doctor"}
        assert set(app.MODES.keys()) == expected

    def test_css_path_set(self):
        from silo.tui.app import create_tui_app

        app = create_tui_app()
        assert app.CSS_PATH == "styles.tcss"


class TestWidgetImports:
    def test_status_counts_import(self):
        from silo.tui.widgets import StatusCounts
        assert StatusCounts is not None

    def test_log_viewer_import(self):
        from silo.tui.widgets import LogViewer
        assert LogViewer is not None

    def test_confirm_modal_import(self):
        from silo.tui.widgets import ConfirmModal
        assert ConfirmModal is not None


class TestScreenImports:
    def test_dashboard_import(self):
        from silo.tui.screens import DashboardScreen
        assert DashboardScreen is not None

    def test_servers_import(self):
        from silo.tui.screens import ServersScreen
        assert ServersScreen is not None

    def test_models_import(self):
        from silo.tui.screens import ModelsScreen
        assert ModelsScreen is not None

    def test_flows_import(self):
        from silo.tui.screens import FlowsScreen
        assert FlowsScreen is not None

    def test_cluster_import(self):
        from silo.tui.screens import ClusterScreen
        assert ClusterScreen is not None

    def test_doctor_import(self):
        from silo.tui.screens import DoctorScreen
        assert DoctorScreen is not None


class TestStatusCountsWidget:
    def test_render_default(self):
        from silo.tui.widgets.status_counts import StatusCounts

        widget = StatusCounts()
        rendered = widget.render()
        assert "Running:" in rendered
        assert "Registered:" in rendered
        assert "Memory:" in rendered

    def test_render_with_values(self):
        from silo.tui.widgets.status_counts import StatusCounts

        widget = StatusCounts()
        widget.running = 3
        widget.registered = 10
        widget.memory_pct = 45.2
        widget.memory_pressure = "normal"
        rendered = widget.render()
        assert "3" in rendered
        assert "10" in rendered
        assert "45%" in rendered
        assert "green" in rendered  # normal = green indicator


class TestLogViewerWidget:
    def test_init_defaults(self):
        from silo.tui.widgets.log_viewer import LogViewer

        viewer = LogViewer()
        assert viewer.log_path is None
        assert viewer._offset == 0

    def test_set_log_path(self):
        from silo.tui.widgets.log_viewer import LogViewer

        viewer = LogViewer()
        test_path = Path("/tmp/test.log")
        viewer._log_path = test_path
        assert viewer.log_path == test_path

    def test_poll_file_no_path(self):
        from silo.tui.widgets.log_viewer import LogViewer

        viewer = LogViewer()
        # Should not raise when path is None
        viewer._poll_file()

    def test_poll_file_nonexistent(self):
        from silo.tui.widgets.log_viewer import LogViewer

        viewer = LogViewer(log_path=Path("/tmp/nonexistent_silo.log"))
        # Should not raise for missing file
        viewer._poll_file()

    def test_poll_file_reads_content(self, tmp_path):
        from silo.tui.widgets.log_viewer import LogViewer

        log_file = tmp_path / "test.log"
        log_file.write_text("line1\nline2\n")

        viewer = LogViewer(log_path=log_file)
        # Mock write to capture output
        written = []
        viewer.write = lambda line: written.append(line)
        viewer._poll_file()

        assert "line1" in written
        assert "line2" in written
        assert viewer._offset > 0

    def test_poll_file_incremental(self, tmp_path):
        from silo.tui.widgets.log_viewer import LogViewer

        log_file = tmp_path / "test.log"
        log_file.write_text("first\n")

        viewer = LogViewer(log_path=log_file)
        written = []
        viewer.write = lambda line: written.append(line)
        viewer._poll_file()
        assert len(written) == 1

        # Append more
        with log_file.open("a") as f:
            f.write("second\n")

        written.clear()
        viewer._poll_file()
        assert len(written) == 1
        assert "second" in written


class TestConfirmModal:
    def test_init(self):
        from silo.tui.widgets.confirm_modal import ConfirmModal

        modal = ConfirmModal("Delete this?")
        assert modal._question == "Delete this?"


@pytest.fixture
def _mock_business_logic(monkeypatch):
    """Mock all business logic for TUI screen tests."""

    @dataclass(frozen=True)
    class FakeProcessInfo:
        name: str = "test-model"
        pid: int = 1234
        port: int = 8800
        repo_id: str = "test-org/test-model"
        status: str = "running"

    @dataclass(frozen=True)
    class FakeMemoryInfo:
        total_gb: float = 32.0
        available_gb: float = 20.0
        used_gb: float = 12.0
        pressure: str = "normal"

        @property
        def usage_percent(self) -> float:
            return (self.used_gb / self.total_gb) * 100

    @dataclass(frozen=True)
    class FakeCheckResult:
        name: str
        status: str
        message: str

    monkeypatch.setattr(
        "silo.process.manager.list_running",
        lambda **kw: [FakeProcessInfo()],
    )
    monkeypatch.setattr(
        "silo.process.manager.get_status",
        lambda name, **kw: FakeProcessInfo(name=name),
    )
    monkeypatch.setattr(
        "silo.process.memory.get_memory_info",
        lambda: FakeMemoryInfo(),
    )
    monkeypatch.setattr(
        "silo.doctor.checks.run_all_checks",
        lambda: [
            FakeCheckResult("python", "ok", "3.12.0"),
            FakeCheckResult("mlx", "ok", "installed"),
            FakeCheckResult("ffmpeg", "warn", "not found"),
        ],
    )

    fake_config = MagicMock()
    fake_model = MagicMock()
    fake_model.name = "test-model"
    fake_model.repo = "test-org/test-model"
    fake_model.host = "127.0.0.1"
    fake_model.port = 8800
    fake_model.quantize = None
    fake_model.output = None
    fake_config.models = [fake_model]
    monkeypatch.setattr(
        "silo.config.loader.load_config",
        lambda **kw: fake_config,
    )

    from silo.registry.models import ModelFormat, RegistryEntry
    from silo.registry.store import Registry

    fake_entry = RegistryEntry(
        repo_id="test-org/test-model",
        format=ModelFormat.MLX,
        size_bytes=4_000_000_000,
    )
    monkeypatch.setattr(
        "silo.registry.store.Registry.load",
        classmethod(lambda cls, **kw: Registry({fake_entry.repo_id: fake_entry})),
    )


@pytest.mark.usefixtures("_mock_business_logic")
class TestDashboardScreenAsync:
    @pytest.mark.asyncio
    async def test_dashboard_mounts(self):
        from silo.tui.screens.dashboard import DashboardScreen
        from textual.app import App

        class TestApp(App):
            MODES = {"dashboard": DashboardScreen}

            def on_mount(self):
                self.switch_mode("dashboard")

        async with TestApp().run_test() as pilot:
            # Dashboard should mount and show status counts
            app = pilot.app
            from silo.tui.widgets.status_counts import StatusCounts

            counts = app.screen.query_one(StatusCounts)
            assert counts is not None


@pytest.mark.usefixtures("_mock_business_logic")
class TestDoctorScreenAsync:
    @pytest.mark.asyncio
    async def test_doctor_mounts(self):
        from silo.tui.screens.doctor import DoctorScreen
        from textual.app import App
        from textual.widgets import DataTable

        class TestApp(App):
            MODES = {"doctor": DoctorScreen}

            def on_mount(self):
                self.switch_mode("doctor")

        async with TestApp().run_test() as pilot:
            table = pilot.app.screen.query_one("#checks-table", DataTable)
            assert table is not None


@pytest.mark.usefixtures("_mock_business_logic")
class TestServersScreenAsync:
    @pytest.mark.asyncio
    async def test_servers_mounts(self):
        from silo.tui.screens.servers import ServersScreen
        from textual.app import App
        from textual.widgets import DataTable

        class TestApp(App):
            MODES = {"servers": ServersScreen}

            def on_mount(self):
                self.switch_mode("servers")

        async with TestApp().run_test() as pilot:
            table = pilot.app.screen.query_one("#server-table", DataTable)
            assert table is not None


class TestModelsScreenAsync:
    @pytest.mark.asyncio
    async def test_models_mounts(self):
        from silo.tui.screens.models import ModelsScreen
        from textual.app import App
        from textual.widgets import DataTable

        class TestApp(App):
            MODES = {"models": ModelsScreen}

            def on_mount(self):
                self.switch_mode("models")

        async with TestApp().run_test() as pilot:
            local_table = pilot.app.screen.query_one("#local-table", DataTable)
            assert local_table is not None
            search_table = pilot.app.screen.query_one(
                "#search-table", DataTable
            )
            assert search_table is not None


@pytest.mark.usefixtures("_mock_business_logic")
class TestClusterScreenAsync:
    @pytest.mark.asyncio
    async def test_cluster_mounts(self):
        from silo.tui.screens.cluster import ClusterScreen
        from textual.app import App
        from textual.widgets import DataTable

        class TestApp(App):
            MODES = {"cluster": ClusterScreen}

            def on_mount(self):
                self.switch_mode("cluster")

        async with TestApp().run_test() as pilot:
            workers_table = pilot.app.screen.query_one(
                "#workers-table", DataTable
            )
            assert workers_table is not None
            models_table = pilot.app.screen.query_one(
                "#cluster-models-table", DataTable
            )
            assert models_table is not None


class TestRegisterModalWidget:
    def test_register_modal_import(self):
        from silo.tui.widgets.register_modal import RegisterModal
        assert RegisterModal is not None


class TestFlowsScreenAsync:
    @pytest.mark.asyncio
    async def test_flows_mounts(self):
        from silo.tui.screens.flows import FlowsScreen
        from textual.app import App
        from textual.widgets import DataTable

        class TestApp(App):
            MODES = {"flows": FlowsScreen}

            def on_mount(self):
                self.switch_mode("flows")

        async with TestApp().run_test() as pilot:
            flow_table = pilot.app.screen.query_one("#flow-table", DataTable)
            assert flow_table is not None
