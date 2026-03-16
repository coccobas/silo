"""MLX model conversion wrapping mlx_lm.convert."""

from __future__ import annotations

from pathlib import Path


def convert_model(
    repo_id: str,
    quantize: str | None = None,
    output: str | None = None,
) -> Path:
    """Convert a model to MLX format.

    Args:
        repo_id: HuggingFace repository ID.
        quantize: Quantization type (e.g., "q4", "q8").
        output: Output directory path.

    Returns:
        Path to the converted model.

    Raises:
        ImportError: If mlx_lm is not installed.
    """
    try:
        from mlx_lm import convert as mlx_convert
    except ImportError as e:
        raise ImportError(
            "mlx-lm is required for model conversion. "
            "Install with: uv pip install 'silo[mlx]'"
        ) from e

    output_path = Path(output) if output else Path.home() / "models" / repo_id.replace("/", "--")

    kwargs: dict[str, object] = {
        "hf_path": repo_id,
        "mlx_path": str(output_path),
    }

    if quantize:
        kwargs["quantize"] = True
        kwargs["q_bits"] = _parse_quantize(quantize)

    mlx_convert(**kwargs)

    return output_path


def _parse_quantize(quantize: str) -> int:
    """Parse quantization string to bit count."""
    mapping = {"q4": 4, "q8": 8, "q2": 2, "q3": 3, "q6": 6}
    q = quantize.lower().strip()
    if q in mapping:
        return mapping[q]
    try:
        return int(q)
    except ValueError:
        raise ValueError(
            f"Invalid quantization '{quantize}'. Use q2, q3, q4, q6, q8, or a number."
        )
