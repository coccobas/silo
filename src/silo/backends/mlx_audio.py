"""MLX Audio backends for STT and TTS."""

from __future__ import annotations

import gc
from collections.abc import Iterator


class MlxAudioSttBackend:
    """Speech-to-text backend wrapping mlx-audio/whisper."""

    def __init__(self) -> None:
        self._model = None
        self._model_path: str | None = None

    def load(self, model_path: str, config: dict) -> None:  # type: ignore[type-arg]
        """Load a Whisper model via mlx_whisper."""
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "mlx-whisper is required for STT. Install with: uv pip install 'silo[audio]'"
            ) from e

        self._model_path = model_path
        self._model = True  # mlx_whisper uses path directly, no separate load step

    def unload(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._model_path = None
        gc.collect()

    def health(self) -> dict:  # type: ignore[type-arg]
        """Return health status."""
        return {
            "status": "ok" if self._model else "unloaded",
            "model": self._model_path or "",
            "backend": "mlx-audio-stt",
        }

    def transcribe(
        self,
        audio: bytes,
        language: str | None = None,
        response_format: str = "json",
        content_type: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        """Transcribe audio using mlx_whisper."""
        import tempfile

        try:
            import mlx_whisper
        except ImportError as e:
            raise ImportError(
                "mlx-whisper is required for STT. Install with: uv pip install 'silo[audio]'"
            ) from e

        if not self._model_path:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Determine file suffix from content type so mlx_whisper picks the right decoder
        suffix = _suffix_from_content_type(content_type) if content_type else ".wav"

        # mlx_whisper expects a file path, so write audio bytes to a temp file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio)
            tmp.flush()

            kwargs: dict = {"path_or_hf_repo": self._model_path}  # type: ignore[type-arg]
            if language:
                kwargs["language"] = language

            result = mlx_whisper.transcribe(tmp.name, **kwargs)

        if response_format == "text":
            return {"text": result.get("text", "")}

        if response_format == "verbose_json":
            return {
                "text": result.get("text", ""),
                "language": result.get("language", ""),
                "duration": result.get("duration", 0.0),
                "segments": result.get("segments", []),
            }

        # Default: json
        return {"text": result.get("text", "")}


class MlxAudioTtsBackend:
    """Text-to-speech backend wrapping mlx-audio.

    Uses the ``mlx_audio.tts.utils.load_model`` / ``model.generate()`` API.
    ``model.generate()`` yields result objects with ``.audio`` (mx.array
    waveform) and ``.sample_rate``.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._model_path: str | None = None

    def load(self, model_path: str, config: dict) -> None:  # type: ignore[type-arg]
        """Load a TTS model via mlx_audio."""
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as e:
            raise ImportError(
                "mlx-audio is required for TTS. Install with: uv pip install 'silo[audio]'"
            ) from e

        self._model = load_model(model_path)
        self._model_path = model_path

    def unload(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._model_path = None
        gc.collect()

    def health(self) -> dict:  # type: ignore[type-arg]
        """Return health status."""
        return {
            "status": "ok" if self._model else "unloaded",
            "model": self._model_path or "",
            "backend": "mlx-audio-tts",
        }

    def voices(self) -> list[dict[str, str]]:
        """Return available voices for the loaded TTS model."""
        # Kokoro voices — the most common mlx-audio TTS model
        # TODO: dynamically discover voices from the loaded model when mlx-audio exposes an API
        return [
            {"id": "af_heart", "name": "Heart (Female)"},
            {"id": "af_bella", "name": "Bella (Female)"},
            {"id": "af_nicole", "name": "Nicole (Female)"},
            {"id": "af_sarah", "name": "Sarah (Female)"},
            {"id": "af_sky", "name": "Sky (Female)"},
            {"id": "am_adam", "name": "Adam (Male)"},
            {"id": "am_michael", "name": "Michael (Male)"},
            {"id": "bf_emma", "name": "Emma (British Female)"},
            {"id": "bf_isabella", "name": "Isabella (British Female)"},
            {"id": "bm_george", "name": "George (British Male)"},
            {"id": "bm_lewis", "name": "Lewis (British Male)"},
        ]

    def speak(
        self,
        text: str,
        voice: str = "af_heart",
        response_format: str = "wav",
        speed: float = 1.0,
        stream: bool = False,
    ) -> bytes | Iterator[bytes]:
        """Generate speech from text."""
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")

        if stream:
            return self._stream_speak(text, voice, speed, response_format)

        return self._generate_full(text, voice, speed, response_format)

    def _generate_full(
        self, text: str, voice: str, speed: float, response_format: str
    ) -> bytes:
        """Generate full audio and encode to requested format."""
        import numpy as np

        chunks: list[object] = []
        sample_rate = 24000

        for result in self._model.generate(text, voice=voice, speed=speed):  # type: ignore[union-attr]
            chunks.append(result.audio)
            sample_rate = getattr(result, "sample_rate", sample_rate)

        # Concatenate all chunks into a single waveform
        audio_np = np.concatenate([np.array(c) for c in chunks])  # type: ignore[arg-type]
        return self._encode_audio(audio_np, sample_rate, response_format)

    def _stream_speak(
        self, text: str, voice: str, speed: float, response_format: str
    ) -> Iterator[bytes]:
        """Stream TTS audio in chunks."""
        import numpy as np

        sample_rate = 24000
        for result in self._model.generate(text, voice=voice, speed=speed):  # type: ignore[union-attr]
            sr = getattr(result, "sample_rate", sample_rate)
            audio_np = np.array(result.audio)
            yield self._encode_audio(audio_np, sr, "wav")

    @staticmethod
    def _encode_audio(
        audio_np: object, sample_rate: int, response_format: str
    ) -> bytes:
        """Encode numpy audio waveform to the requested format."""
        import io
        import wave

        import numpy as np

        audio_np = np.asarray(audio_np)

        # Normalize float to int16 for WAV
        if audio_np.dtype in (np.float32, np.float64):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)

        if response_format == "pcm":
            return audio_np.tobytes()

        # Default: WAV
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_np.tobytes())

        wav_bytes = buffer.getvalue()

        if response_format in ("wav", ""):
            return wav_bytes

        # For mp3/opus/aac/flac, use ffmpeg
        return _ffmpeg_convert(wav_bytes, response_format)


_CONTENT_TYPE_SUFFIX: dict[str, str] = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/mp4": ".mp4",
    "audio/m4a": ".m4a",
    "audio/aac": ".aac",
}


def _suffix_from_content_type(content_type: str) -> str:
    """Map MIME content type to a file suffix for mlx_whisper."""
    # Strip parameters like "; charset=utf-8"
    mime = content_type.split(";")[0].strip().lower()
    return _CONTENT_TYPE_SUFFIX.get(mime, ".wav")


def _ffmpeg_convert(wav_bytes: bytes, output_format: str) -> bytes:
    """Convert WAV bytes to another format using ffmpeg."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", "pipe:0",
                "-f", output_format,
                "-y",
                "pipe:1",
            ],
            input=wav_bytes,
            capture_output=True,
            check=True,
            timeout=30,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"ffmpeg is required for '{output_format}' format. Install ffmpeg."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr.decode()}") from e

    return result.stdout
