"""Tests for HuggingFace download module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from silo.download.hf import download_model, get_model_info, search_models


class TestDownloadModel:
    def test_download(self, mock_hf):
        path = download_model("test-org/test-model")
        assert isinstance(path, Path)
        assert "test-org/test-model" in str(path)


class TestGetModelInfo:
    def test_returns_dict(self, mock_hf):
        info = get_model_info("test-org/test-model")
        assert info["id"] == "test-org/test-model"
        assert info["author"] == "test-org"
        assert info["downloads"] == 1000
        assert isinstance(info["siblings"], list)

    def test_siblings_format(self, mock_hf):
        info = get_model_info("test-org/test-model")
        assert all("rfilename" in s for s in info["siblings"])


class TestSearchModels:
    def test_search(self):
        mock_model = MagicMock()
        mock_model.id = "mlx-community/test"
        mock_model.author = "mlx-community"
        mock_model.downloads = 500
        mock_model.likes = 10
        mock_model.pipeline_tag = "text-generation"
        mock_model.tags = ["mlx"]

        with patch("huggingface_hub.list_models", return_value=[mock_model]):
            results = search_models("test")

        assert len(results) == 1
        assert results[0]["id"] == "mlx-community/test"

    def test_search_mlx_only(self):
        with patch("huggingface_hub.list_models", return_value=[]) as mock_list:
            search_models("llama", mlx_only=True)
            call_kwargs = mock_list.call_args
            assert "mlx" in call_kwargs.kwargs["search"]
