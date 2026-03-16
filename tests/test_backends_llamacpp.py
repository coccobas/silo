"""Tests for llama.cpp backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestLlamaCppBackend:
    def test_init(self):
        from silo.backends.llamacpp import LlamaCppBackend

        backend = LlamaCppBackend()
        assert backend._llm is None
        assert backend._model_path is None

    def test_health_unloaded(self):
        from silo.backends.llamacpp import LlamaCppBackend

        backend = LlamaCppBackend()
        h = backend.health()
        assert h["status"] == "unloaded"
        assert h["backend"] == "llama.cpp"

    def test_load_import_error(self):
        from silo.backends.llamacpp import LlamaCppBackend

        backend = LlamaCppBackend()
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError, match="llama-cpp-python is required"):
                backend.load("/tmp/model.gguf", {})

    def test_load_success(self):
        from silo.backends.llamacpp import LlamaCppBackend

        backend = LlamaCppBackend()
        mock_llama_module = MagicMock()
        mock_llm = MagicMock()
        mock_llama_module.Llama.return_value = mock_llm

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
            backend.load("/tmp/model.gguf", {})

        assert backend._llm is mock_llm
        assert backend._model_path == "/tmp/model.gguf"

    def test_health_loaded(self):
        from silo.backends.llamacpp import LlamaCppBackend

        backend = LlamaCppBackend()
        backend._llm = MagicMock()
        backend._model_path = "/tmp/model.gguf"
        h = backend.health()
        assert h["status"] == "ok"
        assert h["model"] == "/tmp/model.gguf"

    def test_unload(self):
        from silo.backends.llamacpp import LlamaCppBackend

        backend = LlamaCppBackend()
        backend._llm = MagicMock()
        backend._model_path = "/tmp/model.gguf"
        backend.unload()
        assert backend._llm is None
        assert backend._model_path is None

    def test_chat_not_loaded(self):
        from silo.backends.llamacpp import LlamaCppBackend

        backend = LlamaCppBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.chat([{"role": "user", "content": "Hi"}])

    def test_chat_non_streaming(self):
        from silo.backends.llamacpp import LlamaCppBackend

        backend = LlamaCppBackend()
        backend._llm = MagicMock()
        backend._model_path = "/tmp/model.gguf"
        backend._llm.create_chat_completion.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
        }

        result = backend.chat(
            [{"role": "user", "content": "Hi"}],
            stream=False,
            max_tokens=100,
            temperature=0.5,
        )

        assert result["choices"][0]["message"]["content"] == "Hello!"
        backend._llm.create_chat_completion.assert_called_once()

    def test_chat_streaming(self):
        from silo.backends.llamacpp import LlamaCppBackend

        backend = LlamaCppBackend()
        backend._llm = MagicMock()
        backend._model_path = "/tmp/model.gguf"
        backend._llm.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ])

        chunks = list(backend.chat(
            [{"role": "user", "content": "Hi"}],
            stream=True,
        ))

        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert chunks[1]["choices"][0]["delta"]["content"] == " world"

    def test_protocol_compliance(self):
        from silo.backends.llamacpp import LlamaCppBackend
        from silo.backends.protocols import ChatBackend

        assert isinstance(LlamaCppBackend(), ChatBackend)

    def test_load_with_config_options(self):
        from silo.backends.llamacpp import LlamaCppBackend

        backend = LlamaCppBackend()
        mock_llama_module = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
            backend.load("/tmp/model.gguf", {"n_ctx": 8192, "n_gpu_layers": 32})

        call_kwargs = mock_llama_module.Llama.call_args
        assert call_kwargs.kwargs["n_ctx"] == 8192
        assert call_kwargs.kwargs["n_gpu_layers"] == 32


class TestFactoryLlamaCpp:
    def test_gguf_returns_llamacpp(self):
        from silo.backends.factory import resolve_backend
        from silo.backends.llamacpp import LlamaCppBackend
        from silo.registry.models import ModelFormat

        backend = resolve_backend(ModelFormat.GGUF)
        assert isinstance(backend, LlamaCppBackend)

    def test_explicit_llamacpp(self):
        from silo.backends.factory import resolve_backend
        from silo.backends.llamacpp import LlamaCppBackend
        from silo.registry.models import ModelFormat

        backend = resolve_backend(ModelFormat.MLX, backend_override="llamacpp")
        assert isinstance(backend, LlamaCppBackend)
