"""Download tracker — shared state for active downloads displayed in the TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import Lock
from time import time


class DownloadStatus(StrEnum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    DONE = "done"
    FAILED = "failed"


def _fmt_bytes(b: int) -> str:
    """Format byte count as human-readable size."""
    if b >= 1_073_741_824:
        return f"{b / 1_073_741_824:.1f} GB"
    if b >= 1_048_576:
        return f"{b / 1_048_576:.0f} MB"
    if b >= 1024:
        return f"{b / 1024:.0f} KB"
    return f"{b} B"


def _dir_size(path: str) -> int:
    """Get total size of files in a directory (non-recursive is fine for blobs)."""
    try:
        total = 0
        for f in Path(path).rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
    except Exception:
        return 0


@dataclass
class DownloadEntry:
    """A single tracked download."""

    repo_id: str
    node: str = "local"
    status: DownloadStatus = DownloadStatus.PENDING
    started_at: float = field(default_factory=time)
    finished_at: float | None = None
    error: str | None = None
    local_path: str | None = None

    # Progress tracking
    total_bytes: int = 0
    downloaded_bytes: int = 0
    _last_bytes: int = 0
    _last_check: float = 0.0
    _speed_bps: float = 0.0

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time()
        return end - self.started_at

    @property
    def elapsed_str(self) -> str:
        secs = int(self.elapsed)
        if secs < 60:
            return f"{secs}s"
        return f"{secs // 60}m {secs % 60}s"

    @property
    def progress_pct(self) -> float:
        """Download progress as a percentage (0-100)."""
        if self.status == DownloadStatus.DONE:
            return 100.0
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, (self.downloaded_bytes / self.total_bytes) * 100)

    @property
    def progress_str(self) -> str:
        """Human-readable progress string."""
        if self.status == DownloadStatus.DONE:
            return "100%"
        if self.status == DownloadStatus.FAILED:
            return "failed"
        if self.total_bytes <= 0:
            if self.downloaded_bytes > 0:
                return _fmt_bytes(self.downloaded_bytes)
            return "—"
        return f"{self.progress_pct:.0f}%"

    @property
    def speed_str(self) -> str:
        """Human-readable download speed."""
        if self.status != DownloadStatus.DOWNLOADING:
            return "—"
        if self._speed_bps <= 0:
            return "—"
        return f"{_fmt_bytes(int(self._speed_bps))}/s"

    @property
    def eta_str(self) -> str:
        """Estimated time remaining."""
        if self.status != DownloadStatus.DOWNLOADING:
            return "—"
        if self._speed_bps <= 0 or self.total_bytes <= 0:
            return "—"
        remaining = self.total_bytes - self.downloaded_bytes
        if remaining <= 0:
            return "—"
        eta_secs = int(remaining / self._speed_bps)
        if eta_secs < 60:
            return f"{eta_secs}s"
        return f"{eta_secs // 60}m {eta_secs % 60}s"

    def update_progress(self, downloaded: int) -> None:
        """Update downloaded bytes and recalculate speed."""
        now = time()
        dt = now - self._last_check if self._last_check > 0 else 0
        if dt > 0.5:
            delta_bytes = downloaded - self._last_bytes
            # Smoothed speed (exponential moving average)
            new_speed = delta_bytes / dt
            if self._speed_bps > 0:
                self._speed_bps = 0.7 * self._speed_bps + 0.3 * new_speed
            else:
                self._speed_bps = new_speed
            self._last_bytes = downloaded
            self._last_check = now
        self.downloaded_bytes = downloaded


class DownloadTracker:
    """Thread-safe tracker for active and recent downloads."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._downloads: dict[str, DownloadEntry] = {}

    def start(
        self,
        repo_id: str,
        node: str = "local",
        total_bytes: int = 0,
    ) -> DownloadEntry:
        with self._lock:
            entry = DownloadEntry(
                repo_id=repo_id,
                node=node,
                status=DownloadStatus.DOWNLOADING,
                total_bytes=total_bytes,
            )
            self._downloads[repo_id] = entry
            return entry

    def update_progress(self, repo_id: str, downloaded_bytes: int) -> None:
        with self._lock:
            if repo_id in self._downloads:
                self._downloads[repo_id].update_progress(downloaded_bytes)

    def complete(self, repo_id: str, local_path: str) -> None:
        with self._lock:
            if repo_id in self._downloads:
                entry = self._downloads[repo_id]
                entry.status = DownloadStatus.DONE
                entry.finished_at = time()
                entry.local_path = local_path
                if entry.total_bytes > 0:
                    entry.downloaded_bytes = entry.total_bytes

    def fail(self, repo_id: str, error: str) -> None:
        with self._lock:
            if repo_id in self._downloads:
                entry = self._downloads[repo_id]
                entry.status = DownloadStatus.FAILED
                entry.finished_at = time()
                entry.error = error

    def active(self) -> list[DownloadEntry]:
        with self._lock:
            return [
                e for e in self._downloads.values()
                if e.status == DownloadStatus.DOWNLOADING
            ]

    def recent(self, limit: int = 10) -> list[DownloadEntry]:
        with self._lock:
            entries = sorted(
                self._downloads.values(),
                key=lambda e: e.started_at,
                reverse=True,
            )
            return entries[:limit]

    def poll_active_progress(self) -> None:
        """Check filesystem to update progress for active downloads.

        Call this periodically from a timer to update download progress
        by scanning the HF cache directory for downloaded bytes.
        """
        with self._lock:
            for entry in self._downloads.values():
                if entry.status != DownloadStatus.DOWNLOADING:
                    continue
                # Check HF cache for this repo
                cache_dir = self._find_cache_dir(entry.repo_id)
                if cache_dir:
                    size = _dir_size(cache_dir)
                    entry.update_progress(size)

    @staticmethod
    def _find_cache_dir(repo_id: str) -> str | None:
        """Find the HF cache directory for a repo."""
        cache_base = Path.home() / ".cache" / "huggingface" / "hub"
        # HF cache uses -- as separator
        dir_name = f"models--{repo_id.replace('/', '--')}"
        candidate = cache_base / dir_name
        if candidate.exists():
            return str(candidate)
        return None

    def clear_finished(self) -> None:
        with self._lock:
            self._downloads = {
                k: v
                for k, v in self._downloads.items()
                if v.status == DownloadStatus.DOWNLOADING
            }
