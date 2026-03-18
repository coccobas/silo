"""High-level LiteLLM registration facade.

All functions are fire-and-forget: they log warnings on failure but
never raise, so LiteLLM downtime cannot block Silo operations.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from silo.config.models import LitellmConfig
from silo.litellm.client import LitellmClient

logger = logging.getLogger(__name__)

_DEFAULT_LITELLM_PORT = 4000


def normalize_litellm_url(url: str) -> str:
    """Normalize a LiteLLM URL, adding scheme and default port if missing.

    Examples:
        "100.112.188.75"       -> "http://100.112.188.75:4000"
        "100.112.188.75:5000"  -> "http://100.112.188.75:5000"
        "http://10.0.0.1"      -> "http://10.0.0.1:4000"
        "http://10.0.0.1:4000" -> "http://10.0.0.1:4000"
    """
    raw = url.strip().rstrip("/")
    if not raw:
        return ""

    # Add scheme if missing
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"

    parsed = urlparse(raw)
    host = parsed.hostname or ""
    port = parsed.port
    scheme = parsed.scheme or "http"

    if not port:
        port = _DEFAULT_LITELLM_PORT

    return f"{scheme}://{host}:{port}"


def _detect_netbird_fqdn() -> str | None:
    """Try to get the NetBird FQDN from `netbird status`."""
    import subprocess

    try:
        result = subprocess.run(
            ["netbird", "status"],
            capture_output=True, text=True, timeout=3,
        )
        for line in result.stdout.splitlines():
            if line.strip().startswith("FQDN:"):
                fqdn = line.split(":", 1)[1].strip()
                if fqdn:
                    return fqdn
    except Exception:
        pass
    return None


def resolve_api_base(host: str, port: int, advertise_host: str = "") -> str:
    """Build an externally-reachable api_base URL.

    Resolution order:
    1. ``advertise_host`` from config (explicit override)
    2. NetBird FQDN (auto-detected, stable across IP changes)
    3. VPN/LAN IP via ``_detect_local_ip()``

    Only triggers for localhost/wildcard bind addresses.
    """
    resolved = host
    if host in ("127.0.0.1", "0.0.0.0", "localhost", "::1"):
        if advertise_host:
            resolved = advertise_host
        else:
            fqdn = _detect_netbird_fqdn()
            if fqdn:
                resolved = fqdn
            else:
                from silo.agent.daemon import _detect_local_ip

                resolved = _detect_local_ip()
    return f"http://{resolved}:{port}/v1"


def register_model(
    config: LitellmConfig,
    model_name: str,
    host: str,
    port: int,
    instance_id: str,
) -> None:
    """Register a model with LiteLLM.  No-op if integration is disabled."""
    if not config.enabled:
        return
    try:
        client = LitellmClient(config.url, config.api_key)
        api_base = resolve_api_base(host, port, advertise_host=config.advertise_host)
        client.register(model_name, api_base, instance_id)
    except Exception:
        logger.warning("Failed to register '%s' with LiteLLM", model_name, exc_info=True)


def deregister_model(
    config: LitellmConfig,
    model_name: str,
    instance_id: str,
) -> None:
    """Deregister a model from LiteLLM by instance ID.  No-op if disabled."""
    if not config.enabled:
        return
    if not instance_id:
        logger.debug("No instance_id for '%s', skipping LiteLLM deregister", model_name)
        return
    try:
        client = LitellmClient(config.url, config.api_key)
        client.delete(instance_id)
    except Exception:
        logger.warning("Failed to deregister '%s' from LiteLLM", model_name, exc_info=True)


def deregister_all(config: LitellmConfig, api_base_prefix: str = "") -> int:
    """Deregister all models matching an api_base prefix.

    Useful for cleanup on quit — removes all models that belong to
    this Silo instance.  Returns the number of models deregistered.
    """
    if not config.enabled or not config.deregister_on_quit:
        return 0
    try:
        client = LitellmClient(config.url, config.api_key)
        models = client.list_models()
        count = 0
        for entry in models:
            params = entry.get("litellm_params", {})
            entry_base = params.get("api_base", "")
            model_info = entry.get("model_info", {})
            entry_id = model_info.get("id", "")
            if api_base_prefix and not entry_base.startswith(api_base_prefix):
                continue
            if entry_id and client.delete(entry_id):
                count += 1
        if count:
            logger.info("Deregistered %d model(s) from LiteLLM on quit", count)
        return count
    except Exception:
        logger.warning("Failed to deregister models from LiteLLM on quit", exc_info=True)
        return 0
