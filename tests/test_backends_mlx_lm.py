"""Tests for MlxLmBackend with mocked mlx_lm."""

from unittest.mock import MagicMock, patch

import pytest

from silo.backends.mlx_lm import MlxLmBackend


class TestMlxLmBackend:
    def test_init(self):
        backend = MlxLmBackend()
        assert backend._model is None
        assert backend._tokenizer is None
        assert backend._model_path == ""

    def test_health_unloaded(self):
        backend = MlxLmBackend()
        health = backend.health()
        assert health["status"] == "unloaded"
        assert health["backend"] == "mlx-lm"

    def test_load_import_error(self):
        backend = MlxLmBackend()
        with patch.dict("sys.modules", {"mlx_lm": None}):
            with pytest.raises(ImportError, match="mlx-lm is required"):
                backend.load("/tmp/model")

    def test_load_success(self):
        mock_mlx_lm = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        backend = MlxLmBackend()
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            backend.load("/tmp/model")

        assert backend._model is mock_model
        assert backend._tokenizer is mock_tokenizer
        assert backend._model_path == "/tmp/model"

    def test_health_loaded(self):
        backend = MlxLmBackend()
        backend._model = MagicMock()
        backend._model_path = "org/model"
        health = backend.health()
        assert health["status"] == "ok"
        assert health["model"] == "org/model"

    def test_unload(self):
        backend = MlxLmBackend()
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()
        backend._model_path = "org/model"
        backend.unload()
        assert backend._model is None
        assert backend._tokenizer is None
        assert backend._model_path == ""

    def test_chat_not_loaded(self):
        backend = MlxLmBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.chat([{"role": "user", "content": "hi"}])

    def test_chat_non_streaming(self):
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate.return_value = "Hello response"

        backend = MlxLmBackend()
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()
        backend._tokenizer.apply_chat_template.return_value = "formatted prompt"
        backend._model_path = "org/model"

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = backend.chat(
                [{"role": "user", "content": "hi"}], stream=False, max_tokens=100
            )

        assert result["choices"][0]["message"]["content"] == "Hello response"
        assert result["model"] == "org/model"

    def test_chat_no_chat_template(self):
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate.return_value = "response"

        backend = MlxLmBackend()
        backend._model = MagicMock()
        backend._tokenizer = MagicMock(spec=[])  # no apply_chat_template
        backend._model_path = "org/model"

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = backend.chat(
                [{"role": "user", "content": "hi"}], stream=False
            )

        assert result["choices"][0]["message"]["content"] == "response"

    def test_chat_streaming(self):
        mock_mlx_lm = MagicMock()
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = " world"
        mock_mlx_lm.stream_generate.return_value = [chunk1, chunk2]

        backend = MlxLmBackend()
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()
        backend._tokenizer.apply_chat_template.return_value = "prompt"
        backend._model_path = "org/model"

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = list(backend.chat(
                [{"role": "user", "content": "hi"}], stream=True
            ))

        assert len(result) == 2
        assert result[0]["choices"][0]["delta"]["content"] == "Hello"
        assert result[1]["choices"][0]["delta"]["content"] == " world"

    def test_chat_streaming_no_text_attr(self):
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.stream_generate.return_value = ["token1"]

        backend = MlxLmBackend()
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()
        backend._tokenizer.apply_chat_template.return_value = "prompt"
        backend._model_path = "org/model"

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = list(backend.chat(
                [{"role": "user", "content": "hi"}], stream=True
            ))

        assert result[0]["choices"][0]["delta"]["content"] == "token1"

    def test_chat_import_error(self):
        backend = MlxLmBackend()
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()

        with patch.dict("sys.modules", {"mlx_lm": None}):
            with pytest.raises(ImportError, match="mlx-lm is required"):
                backend.chat([{"role": "user", "content": "hi"}])
