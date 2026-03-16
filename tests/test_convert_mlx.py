"""Tests for MLX model conversion."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from silo.convert.mlx import _parse_quantize, convert_model


class TestParseQuantize:
    def test_known_values(self):
        assert _parse_quantize("q4") == 4
        assert _parse_quantize("q8") == 8
        assert _parse_quantize("Q4") == 4

    def test_numeric(self):
        assert _parse_quantize("4") == 4

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid quantization"):
            _parse_quantize("abc")


class TestConvertModel:
    def test_import_error(self):
        with patch.dict("sys.modules", {"mlx_lm": None}):
            with pytest.raises(ImportError, match="mlx-lm is required"):
                convert_model("org/model")

    def test_convert_success(self):
        mock_convert = MagicMock()
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.convert = mock_convert

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = convert_model("org/model", output="/tmp/out")

        assert result == Path("/tmp/out")
        mock_convert.assert_called_once()

    def test_convert_with_quantize(self):
        mock_convert = MagicMock()
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.convert = mock_convert

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            convert_model("org/model", quantize="q4", output="/tmp/out")

        call_kwargs = mock_convert.call_args.kwargs
        assert call_kwargs["quantize"] is True
        assert call_kwargs["q_bits"] == 4
