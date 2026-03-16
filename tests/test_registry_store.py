"""Tests for registry store."""

from pathlib import Path

from silo.registry.models import ModelFormat, RegistryEntry
from silo.registry.store import Registry


class TestRegistry:
    def test_empty(self):
        reg = Registry()
        assert reg.list() == []
        assert reg.get("anything") is None

    def test_add(self):
        entry = RegistryEntry(repo_id="org/model", format=ModelFormat.MLX)
        reg = Registry().add(entry)
        assert reg.get("org/model") is not None
        assert reg.get("org/model").format == ModelFormat.MLX

    def test_add_is_immutable(self):
        entry = RegistryEntry(repo_id="org/model", format=ModelFormat.MLX)
        original = Registry()
        updated = original.add(entry)
        assert original.list() == []
        assert len(updated.list()) == 1

    def test_remove(self):
        entry = RegistryEntry(repo_id="org/model", format=ModelFormat.MLX)
        reg = Registry().add(entry).remove("org/model")
        assert reg.list() == []

    def test_remove_nonexistent(self):
        reg = Registry().remove("nonexistent")
        assert reg.list() == []

    def test_search(self):
        e1 = RegistryEntry(repo_id="mlx-community/llama", format=ModelFormat.MLX)
        e2 = RegistryEntry(repo_id="mlx-community/whisper", format=ModelFormat.MLX)
        e3 = RegistryEntry(repo_id="other/model", format=ModelFormat.STANDARD)
        reg = Registry().add(e1).add(e2).add(e3)
        results = reg.search("mlx")
        assert len(results) == 2

    def test_search_case_insensitive(self):
        entry = RegistryEntry(repo_id="MLX-Community/Model", format=ModelFormat.MLX)
        reg = Registry().add(entry)
        assert len(reg.search("mlx")) == 1

    def test_save_and_load(self, tmp_path: Path):
        path = tmp_path / "registry.json"
        entry = RegistryEntry(
            repo_id="org/model",
            format=ModelFormat.MLX,
            local_path="/tmp/model",
            tags=["test"],
        )
        Registry().add(entry).save(path)

        loaded = Registry.load(path)
        assert len(loaded.list()) == 1
        loaded_entry = loaded.get("org/model")
        assert loaded_entry is not None
        assert loaded_entry.format == ModelFormat.MLX
        assert loaded_entry.local_path == "/tmp/model"
        assert loaded_entry.tags == ["test"]

    def test_load_nonexistent(self, tmp_path: Path):
        reg = Registry.load(tmp_path / "nope.json")
        assert reg.list() == []

    def test_entries_property_returns_copy(self):
        entry = RegistryEntry(repo_id="org/model", format=ModelFormat.MLX)
        reg = Registry().add(entry)
        entries = reg.entries
        entries.clear()
        assert len(reg.entries) == 1
