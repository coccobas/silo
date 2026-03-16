"""Tests for model format detection."""

from silo.registry.detector import detect_model_format
from silo.registry.models import ModelFormat


class TestDetectModelFormat:
    def test_gguf_by_extension(self):
        siblings = [{"rfilename": "model-q4_0.gguf"}]
        assert detect_model_format("org/model", siblings) == ModelFormat.GGUF

    def test_mlx_by_repo_id(self):
        assert detect_model_format("mlx-community/Llama-3.2-1B-4bit") == ModelFormat.MLX

    def test_mlx_by_weights_file(self):
        siblings = [
            {"rfilename": "config.json"},
            {"rfilename": "weights.safetensors"},
        ]
        assert detect_model_format("org/model", siblings) == ModelFormat.MLX

    def test_standard_safetensors(self):
        siblings = [
            {"rfilename": "config.json"},
            {"rfilename": "model.safetensors"},
        ]
        assert detect_model_format("org/model", siblings) == ModelFormat.STANDARD

    def test_unknown_no_siblings(self):
        assert detect_model_format("org/model") == ModelFormat.UNKNOWN

    def test_unknown_no_recognizable_files(self):
        siblings = [{"rfilename": "README.md"}, {"rfilename": "config.json"}]
        assert detect_model_format("org/model", siblings) == ModelFormat.UNKNOWN

    def test_gguf_takes_priority(self):
        siblings = [
            {"rfilename": "model.gguf"},
            {"rfilename": "weights.safetensors"},
        ]
        assert detect_model_format("org/model", siblings) == ModelFormat.GGUF

    def test_stt_by_whisper_keyword(self):
        assert detect_model_format("mlx-community/whisper-large-v3-turbo") == ModelFormat.AUDIO_STT

    def test_stt_by_stt_keyword(self):
        assert detect_model_format("org/speech-to-text-model") == ModelFormat.AUDIO_STT

    def test_tts_by_keyword(self):
        assert detect_model_format("mlx-community/kokoro-tts") == ModelFormat.AUDIO_TTS

    def test_tts_by_text_to_speech(self):
        assert detect_model_format("org/text-to-speech-model") == ModelFormat.AUDIO_TTS

    def test_gguf_priority_over_audio(self):
        siblings = [{"rfilename": "whisper.gguf"}]
        assert detect_model_format("org/whisper-model", siblings) == ModelFormat.GGUF
