"""Tests for the retry utility with exponential backoff."""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, call, patch

import pytest

from silo.agent.retry import RetryConfig, calculate_delay, is_retryable, with_retry


# ── RetryConfig ──────────────────────────────────


class TestRetryConfig:
    def test_defaults(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 0.5
        assert config.max_delay == 10.0

    def test_custom_values(self):
        config = RetryConfig(max_retries=5, base_delay=1.0, max_delay=30.0)
        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0

    def test_frozen(self):
        config = RetryConfig()
        with pytest.raises(AttributeError):
            config.max_retries = 10  # type: ignore[misc]


# ── calculate_delay ──────────────────────────────


class TestCalculateDelay:
    def test_first_attempt_bounded(self):
        config = RetryConfig(base_delay=0.5, max_delay=10.0)
        delay = calculate_delay(0, config)
        # base_delay * 2^0 = 0.5; jitter makes it 0..0.5
        assert 0 <= delay <= 0.5

    def test_exponential_growth(self):
        config = RetryConfig(base_delay=1.0, max_delay=100.0)
        # attempt=0 → max 1.0, attempt=1 → max 2.0, attempt=2 → max 4.0
        for attempt in range(5):
            upper_bound = min(config.base_delay * (2**attempt), config.max_delay)
            delay = calculate_delay(attempt, config)
            assert 0 <= delay <= upper_bound

    def test_capped_at_max(self):
        config = RetryConfig(base_delay=1.0, max_delay=5.0)
        # attempt=10 → base * 2^10 = 1024, but capped at 5.0
        delay = calculate_delay(10, config)
        assert 0 <= delay <= 5.0

    @patch("silo.agent.retry.random.uniform", return_value=0.75)
    def test_jitter_uses_uniform(self, mock_uniform: MagicMock):
        config = RetryConfig(base_delay=2.0, max_delay=100.0)
        delay = calculate_delay(1, config)
        # base * 2^1 = 4.0; uniform(0, 4.0) returns 0.75
        mock_uniform.assert_called_once_with(0, 4.0)
        assert delay == 0.75


# ── is_retryable ─────────────────────────────────


class TestIsRetryable:
    def test_connection_error(self):
        assert is_retryable(ConnectionError("refused")) is True

    def test_connection_refused_error(self):
        assert is_retryable(ConnectionRefusedError()) is True

    def test_timeout_error(self):
        assert is_retryable(TimeoutError("timed out")) is True

    def test_url_error(self):
        err = urllib.error.URLError("connection refused")
        assert is_retryable(err) is True

    def test_http_502(self):
        err = urllib.error.HTTPError(
            url="http://x", code=502, msg="Bad Gateway", hdrs=None, fp=None  # type: ignore[arg-type]
        )
        assert is_retryable(err) is True

    def test_http_503(self):
        err = urllib.error.HTTPError(
            url="http://x", code=503, msg="Unavailable", hdrs=None, fp=None  # type: ignore[arg-type]
        )
        assert is_retryable(err) is True

    def test_http_504(self):
        err = urllib.error.HTTPError(
            url="http://x", code=504, msg="Timeout", hdrs=None, fp=None  # type: ignore[arg-type]
        )
        assert is_retryable(err) is True

    def test_http_400_not_retryable(self):
        err = urllib.error.HTTPError(
            url="http://x", code=400, msg="Bad Request", hdrs=None, fp=None  # type: ignore[arg-type]
        )
        assert is_retryable(err) is False

    def test_http_401_not_retryable(self):
        err = urllib.error.HTTPError(
            url="http://x", code=401, msg="Unauthorized", hdrs=None, fp=None  # type: ignore[arg-type]
        )
        assert is_retryable(err) is False

    def test_http_404_not_retryable(self):
        err = urllib.error.HTTPError(
            url="http://x", code=404, msg="Not Found", hdrs=None, fp=None  # type: ignore[arg-type]
        )
        assert is_retryable(err) is False

    def test_value_error_not_retryable(self):
        assert is_retryable(ValueError("bad")) is False

    def test_runtime_error_not_retryable(self):
        assert is_retryable(RuntimeError("nope")) is False

    def test_os_error_retryable(self):
        assert is_retryable(OSError("network unreachable")) is True


# ── with_retry ───────────────────────────────────


class TestWithRetry:
    @patch("silo.agent.retry.time.sleep")
    def test_succeeds_first_try(self, mock_sleep: MagicMock):
        fn = MagicMock(return_value=42)
        result = with_retry(fn, RetryConfig())
        assert result == 42
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("silo.agent.retry.time.sleep")
    def test_retries_on_transient_then_succeeds(self, mock_sleep: MagicMock):
        fn = MagicMock(side_effect=[ConnectionError(), TimeoutError(), 99])
        result = with_retry(fn, RetryConfig(max_retries=3))
        assert result == 99
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("silo.agent.retry.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep: MagicMock):
        fn = MagicMock(side_effect=ConnectionError("down"))
        with pytest.raises(ConnectionError, match="down"):
            with_retry(fn, RetryConfig(max_retries=3))
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("silo.agent.retry.time.sleep")
    def test_no_retry_on_non_transient(self, mock_sleep: MagicMock):
        fn = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            with_retry(fn, RetryConfig())
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("silo.agent.retry.random.uniform", return_value=0.5)
    @patch("silo.agent.retry.time.sleep")
    def test_exponential_sleep_delays(
        self, mock_sleep: MagicMock, mock_uniform: MagicMock
    ):
        fn = MagicMock(
            side_effect=[ConnectionError(), ConnectionError(), "ok"]
        )
        config = RetryConfig(max_retries=3, base_delay=1.0, max_delay=100.0)
        result = with_retry(fn, config)
        assert result == "ok"
        # attempt 0: uniform(0, 1.0*2^0) = uniform(0, 1.0) → 0.5
        # attempt 1: uniform(0, 1.0*2^1) = uniform(0, 2.0) → 0.5
        assert mock_sleep.call_args_list == [call(0.5), call(0.5)]

    @patch("silo.agent.retry.time.sleep")
    def test_passes_args_and_kwargs(self, mock_sleep: MagicMock):
        fn = MagicMock(return_value="done")
        result = with_retry(fn, RetryConfig(), "a", "b", key="val")
        assert result == "done"
        fn.assert_called_once_with("a", "b", key="val")
