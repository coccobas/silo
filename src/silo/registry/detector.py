"""Detect model format from HuggingFace repository metadata."""

from __future__ import annotations

from silo.registry.models import ModelFormat

# Keywords that indicate STT models
_STT_KEYWORDS = ("whisper", "speech-to-text", "stt", "transcri")
# Keywords that indicate TTS models
_TTS_KEYWORDS = ("tts", "text-to-speech", "kokoro", "bark", "speecht5")


def detect_model_format(
    repo_id: str,
    siblings: list[dict[str, str]] | None = None,
) -> ModelFormat:
    """Detect model format from repo ID and file listing.

    Args:
        repo_id: HuggingFace repository ID (e.g., "mlx-community/Llama-3.2-1B-4bit").
        siblings: List of file dicts with "rfilename" keys from HF model_info.

    Returns:
        Detected ModelFormat.
    """
    filenames = [s.get("rfilename", "") for s in (siblings or [])]
    repo_lower = repo_id.lower()

    # Check for GGUF first
    if any(f.endswith(".gguf") for f in filenames):
        return ModelFormat.GGUF

    # Detect audio models by repo name patterns
    if any(kw in repo_lower for kw in _STT_KEYWORDS):
        return ModelFormat.AUDIO_STT

    if any(kw in repo_lower for kw in _TTS_KEYWORDS):
        return ModelFormat.AUDIO_TTS

    # MLX detection
    if "mlx" in repo_lower:
        return ModelFormat.MLX

    has_safetensors = any(f.endswith(".safetensors") for f in filenames)

    if has_safetensors and any(f in filenames for f in ("weights.safetensors",)):
        return ModelFormat.MLX

    if has_safetensors:
        return ModelFormat.STANDARD

    return ModelFormat.UNKNOWN
