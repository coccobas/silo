"""Retry utility with exponential backoff and jitter.

Designed for transient network errors in RemoteClient HTTP calls.
Uses full jitter strategy to avoid thundering herd.
"""

from __future__ import annotations

import logging
import random
import time
import urllib.error
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})


@dataclass(frozen=True)
class RetryConfig:
    """Immutable retry configuration."""

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 10.0


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate sleep duration using full jitter strategy.

    Formula: random(0, min(base_delay * 2^attempt, max_delay))
    """
    ceiling = min(config.base_delay * (2**attempt), config.max_delay)
    return random.uniform(0, ceiling)


def is_retryable(error: Exception) -> bool:
    """Check whether an error is transient and worth retrying."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in _RETRYABLE_STATUS_CODES
    return isinstance(
        error,
        (ConnectionError, TimeoutError, OSError, urllib.error.URLError),
    )


def with_retry(
    fn: ...,
    config: RetryConfig = RetryConfig(),
    *args: ...,
    **kwargs: ...,
) -> ...:
    """Call *fn* with retry on transient errors.

    Retries up to ``config.max_retries`` times with exponential backoff.
    Non-transient errors are raised immediately.
    """
    last_error: Exception | None = None
    for attempt in range(config.max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not is_retryable(exc):
                raise
            last_error = exc
            if attempt < config.max_retries - 1:
                delay = calculate_delay(attempt, config)
                logger.debug(
                    "Retry %d/%d after %.2fs: %s",
                    attempt + 1,
                    config.max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
    raise last_error  # type: ignore[misc]
