"""Registry store — CRUD with atomic JSON persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from silo.config.paths import REGISTRY_PATH, ensure_dirs
from silo.registry.models import RegistryEntry


class Registry:
    """Immutable registry of local models. Mutations return new state."""

    def __init__(self, entries: dict[str, RegistryEntry] | None = None) -> None:
        self._entries: dict[str, RegistryEntry] = dict(entries or {})

    @property
    def entries(self) -> dict[str, RegistryEntry]:
        return dict(self._entries)

    def list(self) -> list[RegistryEntry]:
        return list(self._entries.values())

    def get(self, repo_id: str) -> RegistryEntry | None:
        return self._entries.get(repo_id)

    def add(self, entry: RegistryEntry) -> Registry:
        new_entries = {**self._entries, entry.repo_id: entry}
        return Registry(new_entries)

    def remove(self, repo_id: str, delete_files: bool = False) -> Registry:
        if delete_files:
            entry = self._entries.get(repo_id)
            if entry and entry.local_path:
                import shutil

                path = Path(entry.local_path)
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
        new_entries = {k: v for k, v in self._entries.items() if k != repo_id}
        return Registry(new_entries)

    def search(self, query: str) -> list[RegistryEntry]:
        q = query.lower()
        return [e for e in self._entries.values() if q in e.repo_id.lower()]

    def save(self, path: Path | None = None) -> None:
        """Atomically persist registry to JSON."""
        target = path or REGISTRY_PATH
        ensure_dirs()

        data = {
            repo_id: entry.model_dump() for repo_id, entry in self._entries.items()
        }

        dir_path = target.parent
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.rename(tmp_path, target)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @classmethod
    def load(cls, path: Path | None = None) -> Registry:
        """Load registry from JSON file."""
        target = path or REGISTRY_PATH

        if not target.exists():
            return cls()

        with open(target) as f:
            data = json.load(f)

        entries = {
            repo_id: RegistryEntry.model_validate(entry_data)
            for repo_id, entry_data in data.items()
        }
        return cls(entries)
