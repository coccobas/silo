"""Backend factory — resolve the correct backend for a model format."""

from __future__ import annotations

from silo.backends.protocols import BaseBackend
from silo.registry.models import ModelFormat


def resolve_backend(
    model_format: ModelFormat,
    backend_override: str | None = None,
) -> BaseBackend:
    """Resolve and return the appropriate backend for a model.

    Args:
        model_format: Detected format of the model.
        backend_override: Explicit backend name (e.g., "mlx", "llamacpp").

    Returns:
        An unloaded backend instance.

    Raises:
        ValueError: If the format/backend combination is unsupported.
    """
    override = (backend_override or "").lower()

    if override == "llamacpp" or (not override and model_format == ModelFormat.GGUF):
        from silo.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend()  # type: ignore[return-value]

    if model_format == ModelFormat.AUDIO_STT:
        from silo.backends.mlx_audio import MlxAudioSttBackend

        return MlxAudioSttBackend()  # type: ignore[return-value]

    if model_format == ModelFormat.AUDIO_TTS:
        from silo.backends.mlx_audio import MlxAudioTtsBackend

        return MlxAudioTtsBackend()  # type: ignore[return-value]

    if override in ("mlx", "") and model_format in (ModelFormat.MLX, ModelFormat.STANDARD):
        from silo.backends.mlx_lm import MlxLmBackend

        return MlxLmBackend()  # type: ignore[return-value]

    if override == "mlx":
        from silo.backends.mlx_lm import MlxLmBackend

        return MlxLmBackend()  # type: ignore[return-value]

    raise ValueError(
        f"No backend available for format '{model_format.value}'"
        + (f" with override '{backend_override}'" if backend_override else "")
    )
