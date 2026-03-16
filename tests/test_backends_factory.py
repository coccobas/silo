"""Tests for backend factory."""

import pytest

from silo.backends.factory import resolve_backend
from silo.registry.models import ModelFormat


class TestResolveBackend:
    def test_mlx_format(self):
        backend = resolve_backend(ModelFormat.MLX)
        from silo.backends.mlx_lm import MlxLmBackend

        assert isinstance(backend, MlxLmBackend)

    def test_standard_format(self):
        backend = resolve_backend(ModelFormat.STANDARD)
        from silo.backends.mlx_lm import MlxLmBackend

        assert isinstance(backend, MlxLmBackend)

    def test_explicit_mlx(self):
        backend = resolve_backend(ModelFormat.MLX, backend_override="mlx")
        from silo.backends.mlx_lm import MlxLmBackend

        assert isinstance(backend, MlxLmBackend)

    def test_gguf_returns_llamacpp(self):
        from silo.backends.llamacpp import LlamaCppBackend

        backend = resolve_backend(ModelFormat.GGUF)
        assert isinstance(backend, LlamaCppBackend)

    def test_explicit_llamacpp(self):
        from silo.backends.llamacpp import LlamaCppBackend

        backend = resolve_backend(ModelFormat.MLX, backend_override="llamacpp")
        assert isinstance(backend, LlamaCppBackend)

    def test_unknown_format(self):
        with pytest.raises(ValueError, match="No backend available"):
            resolve_backend(ModelFormat.UNKNOWN)

    def test_audio_stt_format(self):
        backend = resolve_backend(ModelFormat.AUDIO_STT)
        from silo.backends.mlx_audio import MlxAudioSttBackend

        assert isinstance(backend, MlxAudioSttBackend)

    def test_audio_tts_format(self):
        backend = resolve_backend(ModelFormat.AUDIO_TTS)
        from silo.backends.mlx_audio import MlxAudioTtsBackend

        assert isinstance(backend, MlxAudioTtsBackend)
