"""HTTP client for the LiteLLM proxy management API.

All methods are error-safe: they catch exceptions, log warnings,
and return safe defaults so that LiteLLM issues never block Silo.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

_TIMEOUT = 5  # seconds


class LitellmClient:
    """Thin wrapper around the LiteLLM proxy management endpoints."""

    def __init__(self, url: str, api_key: str = "") -> None:
        self._base = url.rstrip("/")
        self._api_key = api_key

    # ── Public API ──────────────────────────────────

    def health(self) -> bool:
        """Return True if the proxy is reachable."""
        resp = self._get("/health")
        return resp is not None

    def register(self, model_name: str, api_base: str, instance_id: str) -> bool:
        """Register a model deployment.  Returns True on success."""
        payload = {
            "model_name": model_name,
            "litellm_params": {
                "model": f"openai/{model_name}",
                "api_base": api_base,
                "api_key": "unused",
            },
            "model_info": {
                "id": instance_id,
            },
        }
        resp = self._post("/model/new", payload)
        if resp is not None:
            logger.info(
                "Registered '%s' with LiteLLM (instance=%s, api_base=%s)",
                model_name, instance_id, api_base,
            )
            return True
        return False

    def delete(self, instance_id: str) -> bool:
        """Delete a model deployment by instance ID.  Returns True on success."""
        resp = self._post("/model/delete", {"id": instance_id})
        if resp is not None:
            logger.info("Deregistered instance %s from LiteLLM", instance_id)
            return True
        return False

    def list_models(self) -> list[dict]:
        """Return all registered model deployments."""
        resp = self._get("/model/info")
        if resp is None:
            return []
        # LiteLLM returns {"data": [...]} for model info
        if isinstance(resp, dict):
            return resp.get("data", [])
        return []

    # ── Internal helpers ────────────────────────────

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _get(self, path: str) -> dict | list | None:
        url = f"{self._base}{path}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read())
        except Exception:
            logger.warning("LiteLLM GET %s failed", url, exc_info=True)
            return None

    def _post(self, path: str, data: dict) -> dict | list | None:
        url = f"{self._base}{path}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body, headers=self._headers(), method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read())
        except Exception:
            logger.warning("LiteLLM POST %s failed", url, exc_info=True)
            return None
