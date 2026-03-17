"""Log viewer widget — tails a log file with periodic polling."""

from __future__ import annotations

from pathlib import Path

from textual.widgets import RichLog


class LogViewer(RichLog):
    """RichLog that tails a file, polling every 0.5 s."""

    DEFAULT_CSS = """
    LogViewer {
        height: 10;
        border-top: solid $accent;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        *,
        log_path: Path | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, wrap=True, highlight=True, markup=True)
        self._log_path: Path | None = log_path
        self._offset: int = 0
        self._timer = None

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    @log_path.setter
    def log_path(self, path: Path | None) -> None:
        self._log_path = path
        self.clear()
        if path is not None and path.exists():
            # Show last N lines, then tail from there
            try:
                text = path.read_text(errors="replace")
                lines = text.splitlines()
                tail = lines[-50:] if len(lines) > 50 else lines
                for line in tail:
                    self.write(line)
                self._offset = path.stat().st_size
            except OSError:
                self._offset = 0
        else:
            self._offset = 0

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.5, self._poll_file)

    def _poll_file(self) -> None:
        if self._log_path is None or not self._log_path.exists():
            return
        try:
            with self._log_path.open("rb") as fh:
                fh.seek(self._offset)
                new_data = fh.read()
                if new_data:
                    self._offset += len(new_data)
                    from datetime import datetime

                    ts = datetime.now().strftime("%H:%M:%S")
                    for line in new_data.decode(errors="replace").splitlines():
                        self.write(f"[dim]{ts}[/] {line}")
        except OSError:
            pass
