"""Prometheus-compatible metrics for model serving."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ModelMetrics:
    """Thread-safe metrics collector for a single model."""

    model_name: str
    _lock: Lock = field(default_factory=Lock, repr=False)
    _request_count: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)
    _error_count: int = field(default=0, repr=False)
    _total_tokens: int = field(default=0, repr=False)
    _total_duration: float = field(default=0.0, repr=False)
    _request_durations: list[float] = field(default_factory=list, repr=False)

    def record_request(self, endpoint: str, status: int, duration: float) -> None:
        """Record a completed request."""
        with self._lock:
            key = f"{endpoint}:{status}"
            self._request_count[key] += 1
            self._request_durations.append(duration)
            if status >= 400:
                self._error_count += 1

    def record_tokens(self, count: int) -> None:
        """Record tokens generated."""
        with self._lock:
            self._total_tokens += count

    def record_duration(self, duration: float) -> None:
        """Record inference duration for tokens/sec calculation."""
        with self._lock:
            self._total_duration += duration

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        with self._lock:
            lines: list[str] = []
            model = self.model_name

            # Request counts
            for key, count in self._request_count.items():
                endpoint, status = key.rsplit(":", 1)
                lines.append(
                    f'silo_requests_total{{model="{model}",'
                    f'endpoint="{endpoint}",status="{status}"}} {count}'
                )

            # Error count
            lines.append(f'silo_errors_total{{model="{model}"}} {self._error_count}')

            # Total tokens
            lines.append(f'silo_tokens_total{{model="{model}"}} {self._total_tokens}')

            # Tokens per second
            tps = self._total_tokens / self._total_duration if self._total_duration > 0 else 0
            lines.append(f'silo_tokens_per_second{{model="{model}"}} {tps:.1f}')

            # Request duration quantiles (p50, p95, p99)
            if self._request_durations:
                sorted_d = sorted(self._request_durations)
                n = len(sorted_d)
                p50 = sorted_d[int(n * 0.5)]
                p95 = sorted_d[min(int(n * 0.95), n - 1)]
                p99 = sorted_d[min(int(n * 0.99), n - 1)]
                lines.append(
                    f'silo_request_duration_seconds{{model="{model}",'
                    f'quantile="0.5"}} {p50:.3f}'
                )
                lines.append(
                    f'silo_request_duration_seconds{{model="{model}",'
                    f'quantile="0.95"}} {p95:.3f}'
                )
                lines.append(
                    f'silo_request_duration_seconds{{model="{model}",'
                    f'quantile="0.99"}} {p99:.3f}'
                )

            # Model loaded
            loaded = 1 if self._total_tokens > 0 or self._request_count else 0
            lines.append(f'silo_model_loaded{{model="{model}"}} {loaded}')

            return "\n".join(lines) + "\n"
