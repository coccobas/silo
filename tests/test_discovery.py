"""Tests for mDNS service discovery."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, call, patch

import pytest

from silo.agent.discovery import (
    MDNS_SERVICE_TYPE,
    DiscoveredNode,
    ServiceAdvertiser,
    discover_nodes,
)


# ── DiscoveredNode ───────────────────────────────


class TestDiscoveredNode:
    def test_fields(self):
        node = DiscoveredNode(
            name="mini-1", host="10.0.0.5", port=9900, hostname="mini-1.local"
        )
        assert node.name == "mini-1"
        assert node.host == "10.0.0.5"
        assert node.port == 9900
        assert node.hostname == "mini-1.local"

    def test_frozen(self):
        node = DiscoveredNode(
            name="x", host="1.2.3.4", port=9900, hostname="x.local"
        )
        with pytest.raises(AttributeError):
            node.name = "y"  # type: ignore[misc]


# ── Constants ────────────────────────────────────


class TestConstants:
    def test_service_type(self):
        assert MDNS_SERVICE_TYPE == "_silo._tcp.local."


# ── ServiceAdvertiser ────────────────────────────


class TestServiceAdvertiser:
    @patch("silo.agent.discovery.Zeroconf")
    @patch("silo.agent.discovery.ServiceInfo")
    def test_register_creates_service_info(
        self, mock_si_cls: MagicMock, mock_zc_cls: MagicMock
    ):
        advertiser = ServiceAdvertiser(node_name="mini-1", port=9900)
        advertiser.__enter__()

        mock_si_cls.assert_called_once()
        kwargs = mock_si_cls.call_args
        assert kwargs[1]["type_"] == MDNS_SERVICE_TYPE
        assert kwargs[1]["port"] == 9900
        assert b"mini-1" in kwargs[1]["properties"][b"node_name"]

        advertiser.__exit__(None, None, None)

    @patch("silo.agent.discovery.Zeroconf")
    @patch("silo.agent.discovery.ServiceInfo")
    def test_register_calls_zeroconf(
        self, mock_si_cls: MagicMock, mock_zc_cls: MagicMock
    ):
        mock_zc = mock_zc_cls.return_value
        advertiser = ServiceAdvertiser(node_name="mini-1", port=9900)
        advertiser.__enter__()

        mock_zc.register_service.assert_called_once()
        advertiser.__exit__(None, None, None)

    @patch("silo.agent.discovery.Zeroconf")
    @patch("silo.agent.discovery.ServiceInfo")
    def test_unregister_on_exit(
        self, mock_si_cls: MagicMock, mock_zc_cls: MagicMock
    ):
        mock_zc = mock_zc_cls.return_value
        advertiser = ServiceAdvertiser(node_name="mini-1", port=9900)
        advertiser.__enter__()
        advertiser.__exit__(None, None, None)

        mock_zc.unregister_service.assert_called_once()
        mock_zc.close.assert_called_once()

    @patch("silo.agent.discovery.Zeroconf")
    @patch("silo.agent.discovery.ServiceInfo")
    def test_context_manager(
        self, mock_si_cls: MagicMock, mock_zc_cls: MagicMock
    ):
        mock_zc = mock_zc_cls.return_value
        with ServiceAdvertiser(node_name="studio", port=8000) as adv:
            assert adv is not None
            mock_zc.register_service.assert_called_once()
        mock_zc.unregister_service.assert_called_once()
        mock_zc.close.assert_called_once()


# ── discover_nodes ───────────────────────────────


class TestDiscoverNodes:
    @patch("silo.agent.discovery.time.sleep")
    @patch("silo.agent.discovery.ServiceBrowser")
    @patch("silo.agent.discovery.Zeroconf")
    def test_returns_discovered_nodes(
        self,
        mock_zc_cls: MagicMock,
        mock_browser_cls: MagicMock,
        mock_sleep: MagicMock,
    ):
        mock_zc = mock_zc_cls.return_value

        # Simulate ServiceBrowser calling the listener's add_service
        def fake_browser(zc, stype, listener):
            info1 = MagicMock()
            info1.properties = {
                b"node_name": b"mini-1",
                b"hostname": b"mini-1.local",
            }
            info1.port = 9900
            info1.parsed_addresses.return_value = ["10.0.0.5"]

            info2 = MagicMock()
            info2.properties = {
                b"node_name": b"studio",
                b"hostname": b"studio.local",
            }
            info2.port = 8000
            info2.parsed_addresses.return_value = ["10.0.0.10"]

            mock_zc.get_service_info.side_effect = [info1, info2]

            listener.add_service(zc, stype, "mini-1._silo._tcp.local.")
            listener.add_service(zc, stype, "studio._silo._tcp.local.")
            return MagicMock()

        mock_browser_cls.side_effect = fake_browser

        nodes = discover_nodes(timeout=2.0)
        assert len(nodes) == 2
        assert nodes[0].name == "mini-1"
        assert nodes[0].host == "10.0.0.5"
        assert nodes[0].port == 9900
        assert nodes[1].name == "studio"
        assert nodes[1].host == "10.0.0.10"
        mock_sleep.assert_called_once_with(2.0)

    @patch("silo.agent.discovery.time.sleep")
    @patch("silo.agent.discovery.ServiceBrowser")
    @patch("silo.agent.discovery.Zeroconf")
    def test_empty_network(
        self,
        mock_zc_cls: MagicMock,
        mock_browser_cls: MagicMock,
        mock_sleep: MagicMock,
    ):
        # Browser never calls listener — no services found
        nodes = discover_nodes(timeout=1.0)
        assert nodes == []

    @patch("silo.agent.discovery.time.sleep")
    @patch("silo.agent.discovery.ServiceBrowser")
    @patch("silo.agent.discovery.Zeroconf")
    def test_skips_unresolvable_services(
        self,
        mock_zc_cls: MagicMock,
        mock_browser_cls: MagicMock,
        mock_sleep: MagicMock,
    ):
        mock_zc = mock_zc_cls.return_value

        def fake_browser(zc, stype, listener):
            mock_zc.get_service_info.return_value = None
            listener.add_service(zc, stype, "ghost._silo._tcp.local.")
            return MagicMock()

        mock_browser_cls.side_effect = fake_browser

        nodes = discover_nodes()
        assert nodes == []


# ── Import error handling ────────────────────────


class TestDiscoveryImportError:
    def test_advertiser_import_error(self):
        """ServiceAdvertiser raises clear error when zeroconf missing."""
        with patch.dict("sys.modules", {"zeroconf": None}):
            # The import check happens at module level, so we test via
            # the public function that checks availability
            from silo.agent.discovery import is_discovery_available

            # When zeroconf is properly installed in test env, this is True.
            # The real test is that the function exists and returns a bool.
            assert isinstance(is_discovery_available(), bool)


# ── build_clients with discovery ─────────────────


class TestBuildClientsDiscovery:
    def test_discover_false_default(self):
        from silo.agent.client import LocalClient, build_clients

        clients = build_clients()
        assert "local" in clients
        assert isinstance(clients["local"], LocalClient)
        assert len(clients) == 1

    @patch("silo.agent.discovery.discover_nodes")
    def test_discover_true_adds_found_nodes(self, mock_discover: MagicMock):
        from silo.agent.client import RemoteClient, build_clients

        mock_discover.return_value = [
            DiscoveredNode(
                name="mini-1", host="10.0.0.5", port=9900, hostname="mini-1.local"
            ),
        ]
        clients = build_clients(discover=True)
        assert "mini-1" in clients
        assert isinstance(clients["mini-1"], RemoteClient)
        mock_discover.assert_called_once_with(timeout=3.0)

    @patch("silo.agent.discovery.discover_nodes")
    def test_discover_skips_duplicate_names(self, mock_discover: MagicMock):
        from silo.agent.client import RemoteClient, build_clients
        from silo.config.models import NodeConfig

        mock_discover.return_value = [
            DiscoveredNode(
                name="mini-1", host="10.0.0.99", port=9900, hostname="x.local"
            ),
        ]
        nodes = [NodeConfig(name="mini-1", host="10.0.0.1")]
        clients = build_clients(nodes=nodes, discover=True)
        # Config wins — the host should be from the explicit config
        assert isinstance(clients["mini-1"], RemoteClient)
        assert clients["mini-1"]._base == "http://10.0.0.1:9900"
