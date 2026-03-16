"""Tests for MLX Audio STT and TTS backends."""

from __future__ import annotations

import gc
from unittest.mock import MagicMock, patch

import pytest


class TestMlxAudioSttBackend:
    def test_init(self):
        from silo.backends.mlx_audio import MlxAudioSttBackend

        backend = MlxAudioSttBackend()
        assert backend._model is None
        assert backend._model_path is None

    def test_health_unloaded(self):
        from silo.backends.mlx_audio import MlxAudioSttBackend

        backend = MlxAudioSttBackend()
        h = backend.health()
        assert h["status"] == "unloaded"
        assert h["backend"] == "mlx-audio-stt"

    def test_load_success(self):
        from silo.backends.mlx_audio import MlxAudioSttBackend

        backend = MlxAudioSttBackend()
        with patch.dict("sys.modules", {"mlx_whisper": MagicMock()}):
            backend.load("/tmp/whisper", {})
        assert backend._model is True
        assert backend._model_path == "/tmp/whisper"

    def test_load_import_error(self):
        from silo.backends.mlx_audio import MlxAudioSttBackend

        backend = MlxAudioSttBackend()
        with patch.dict("sys.modules", {"mlx_whisper": None}):
            with pytest.raises(ImportError, match="mlx-whisper is required"):
                backend.load("/tmp/whisper", {})

    def test_health_loaded(self):
        from silo.backends.mlx_audio import MlxAudioSttBackend

        backend = MlxAudioSttBackend()
        with patch.dict("sys.modules", {"mlx_whisper": MagicMock()}):
            backend.load("/tmp/whisper", {})
        h = backend.health()
        assert h["status"] == "ok"
        assert h["model"] == "/tmp/whisper"

    def test_unload(self):
        from silo.backends.mlx_audio import MlxAudioSttBackend

        backend = MlxAudioSttBackend()
        with patch.dict("sys.modules", {"mlx_whisper": MagicMock()}):
            backend.load("/tmp/whisper", {})
        backend.unload()
        assert backend._model is None
        assert backend._model_path is None

    def test_transcribe_not_loaded(self):
        from silo.backends.mlx_audio import MlxAudioSttBackend

        backend = MlxAudioSttBackend()
        mock_whisper = MagicMock()
        with patch.dict("sys.modules", {"mlx_whisper": mock_whisper}):
            with pytest.raises(RuntimeError, match="not loaded"):
                backend.transcribe(b"audio data")

    def test_transcribe_json(self):
        from silo.backends.mlx_audio import MlxAudioSttBackend

        backend = MlxAudioSttBackend()
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {"text": "Hello world"}

        with patch.dict("sys.modules", {"mlx_whisper": mock_whisper}):
            backend.load("/tmp/whisper", {})
            result = backend.transcribe(b"audio data", response_format="json")

        assert result == {"text": "Hello world"}

    def test_transcribe_text_format(self):
        from silo.backends.mlx_audio import MlxAudioSttBackend

        backend = MlxAudioSttBackend()
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {"text": "Hello world"}

        with patch.dict("sys.modules", {"mlx_whisper": mock_whisper}):
            backend.load("/tmp/whisper", {})
            result = backend.transcribe(b"audio data", response_format="text")

        assert result == {"text": "Hello world"}

    def test_transcribe_verbose_json(self):
        from silo.backends.mlx_audio import MlxAudioSttBackend

        backend = MlxAudioSttBackend()
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
            "duration": 2.5,
            "segments": [{"start": 0.0, "end": 2.5, "text": "Hello world"}],
        }

        with patch.dict("sys.modules", {"mlx_whisper": mock_whisper}):
            backend.load("/tmp/whisper", {})
            result = backend.transcribe(b"audio data", response_format="verbose_json")

        assert result["text"] == "Hello world"
        assert result["language"] == "en"
        assert result["duration"] == 2.5
        assert len(result["segments"]) == 1

    def test_transcribe_with_language(self):
        from silo.backends.mlx_audio import MlxAudioSttBackend

        backend = MlxAudioSttBackend()
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {"text": "Bonjour"}

        with patch.dict("sys.modules", {"mlx_whisper": mock_whisper}):
            backend.load("/tmp/whisper", {})
            result = backend.transcribe(b"audio data", language="fr")

        assert result == {"text": "Bonjour"}
        # Verify language was passed to mlx_whisper
        call_kwargs = mock_whisper.transcribe.call_args
        assert call_kwargs.kwargs.get("language") == "fr"

    def test_protocol_compliance(self):
        from silo.backends.mlx_audio import MlxAudioSttBackend
        from silo.backends.protocols import SttBackend

        assert isinstance(MlxAudioSttBackend(), SttBackend)


class TestMlxAudioTtsBackend:
    def test_init(self):
        from silo.backends.mlx_audio import MlxAudioTtsBackend

        backend = MlxAudioTtsBackend()
        assert backend._model is None
        assert backend._model_path is None

    def test_health_unloaded(self):
        from silo.backends.mlx_audio import MlxAudioTtsBackend

        backend = MlxAudioTtsBackend()
        h = backend.health()
        assert h["status"] == "unloaded"
        assert h["backend"] == "mlx-audio-tts"

    def test_load_success(self):
        from silo.backends.mlx_audio import MlxAudioTtsBackend

        backend = MlxAudioTtsBackend()
        mock_model = MagicMock()
        mock_load_model = MagicMock(return_value=mock_model)

        mock_utils = MagicMock()
        mock_utils.load_model = mock_load_model

        with patch.dict("sys.modules", {
            "mlx_audio": MagicMock(),
            "mlx_audio.tts": MagicMock(),
            "mlx_audio.tts.utils": mock_utils,
        }):
            backend.load("/tmp/tts", {})

        assert backend._model is mock_model
        assert backend._model_path == "/tmp/tts"

    def test_load_import_error(self):
        from silo.backends.mlx_audio import MlxAudioTtsBackend

        backend = MlxAudioTtsBackend()
        with patch.dict("sys.modules", {
            "mlx_audio": None,
            "mlx_audio.tts": None,
            "mlx_audio.tts.utils": None,
        }):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                backend.load("/tmp/tts", {})

    def test_health_loaded(self):
        from silo.backends.mlx_audio import MlxAudioTtsBackend

        backend = MlxAudioTtsBackend()
        backend._model = MagicMock()
        backend._model_path = "/tmp/tts"
        h = backend.health()
        assert h["status"] == "ok"
        assert h["model"] == "/tmp/tts"

    def test_unload(self):
        from silo.backends.mlx_audio import MlxAudioTtsBackend

        backend = MlxAudioTtsBackend()
        backend._model = MagicMock()
        backend._model_path = "/tmp/tts"
        backend.unload()
        assert backend._model is None
        assert backend._model_path is None

    def test_speak_not_loaded(self):
        from silo.backends.mlx_audio import MlxAudioTtsBackend

        backend = MlxAudioTtsBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.speak("Hello world")

    def test_speak_generates_wav(self):
        import numpy as np

        from silo.backends.mlx_audio import MlxAudioTtsBackend

        backend = MlxAudioTtsBackend()

        # Mock the model's generate() method to yield a result
        mock_result = MagicMock()
        mock_result.audio = np.zeros(1000, dtype=np.float32)
        mock_result.sample_rate = 24000
        mock_model = MagicMock()
        mock_model.generate.return_value = iter([mock_result])

        backend._model = mock_model
        backend._model_path = "/tmp/tts"

        result = backend.speak("Hello world", voice="af_heart", response_format="wav")

        assert isinstance(result, bytes)
        # WAV files start with RIFF header
        assert result[:4] == b"RIFF"
        mock_model.generate.assert_called_once_with(
            "Hello world", voice="af_heart", speed=1.0
        )

    def test_speak_stream(self):
        import numpy as np

        from silo.backends.mlx_audio import MlxAudioTtsBackend

        backend = MlxAudioTtsBackend()

        mock_result1 = MagicMock()
        mock_result1.audio = np.zeros(500, dtype=np.float32)
        mock_result1.sample_rate = 24000
        mock_result2 = MagicMock()
        mock_result2.audio = np.zeros(500, dtype=np.float32)
        mock_result2.sample_rate = 24000
        mock_model = MagicMock()
        mock_model.generate.return_value = iter([mock_result1, mock_result2])

        backend._model = mock_model
        backend._model_path = "/tmp/tts"

        chunks = list(backend.speak("Hello", stream=True))
        assert len(chunks) == 2
        for chunk in chunks:
            assert isinstance(chunk, bytes)
            assert chunk[:4] == b"RIFF"

    def test_protocol_compliance(self):
        from silo.backends.mlx_audio import MlxAudioTtsBackend
        from silo.backends.protocols import TtsBackend

        assert isinstance(MlxAudioTtsBackend(), TtsBackend)


class TestFfmpegConvert:
    def test_ffmpeg_not_found(self):
        from silo.backends.mlx_audio import _ffmpeg_convert

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(RuntimeError, match="ffmpeg is required"):
                _ffmpeg_convert(b"wav data", "mp3")

    def test_ffmpeg_error(self):
        import subprocess

        from silo.backends.mlx_audio import _ffmpeg_convert

        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr=b"error msg"),
        ):
            with pytest.raises(RuntimeError, match="ffmpeg conversion failed"):
                _ffmpeg_convert(b"wav data", "mp3")

    def test_ffmpeg_success(self):
        from silo.backends.mlx_audio import _ffmpeg_convert

        mock_result = MagicMock()
        mock_result.stdout = b"converted audio"
        with patch("subprocess.run", return_value=mock_result):
            result = _ffmpeg_convert(b"wav data", "mp3")
        assert result == b"converted audio"
