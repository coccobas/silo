"""llama.cpp backend for GGUF models via llama-cpp-python."""

from __future__ import annotations

import gc
from collections.abc import Iterator


class LlamaCppBackend:
    """Chat backend wrapping llama-cpp-python for GGUF model inference."""

    def __init__(self) -> None:
        self._llm = None
        self._model_path: str | None = None

    def load(self, model_path: str, config: dict) -> None:  # type: ignore[type-arg]
        """Load a GGUF model via llama-cpp-python.

        Args:
            model_path: Path to the .gguf file or HF repo with GGUF files.
            config: Additional configuration (n_ctx, n_gpu_layers, etc.).
        """
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: uv pip install 'silo[llamacpp]'"
            ) from e

        n_ctx = config.get("n_ctx", 4096)
        n_gpu_layers = config.get("n_gpu_layers", -1)  # -1 = all layers on GPU

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self._model_path = model_path

    def unload(self) -> None:
        """Unload model from memory."""
        self._llm = None
        self._model_path = None
        gc.collect()

    def health(self) -> dict:  # type: ignore[type-arg]
        """Return health status."""
        return {
            "status": "ok" if self._llm else "unloaded",
            "model": self._model_path or "",
            "backend": "llama.cpp",
        }

    def chat(
        self,
        messages: list[dict],  # type: ignore[type-arg]
        stream: bool = False,
        **kwargs: object,
    ) -> Iterator[dict] | dict:  # type: ignore[type-arg]
        """OpenAI-compatible chat completion via llama.cpp.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            stream: If True, return an iterator of chunks.
            **kwargs: Additional params (max_tokens, temperature, etc.).

        Returns:
            Complete response dict or iterator of chunk dicts.
        """
        if not self._llm:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_tokens = int(kwargs.get("max_tokens", 512))
        temperature = float(kwargs.get("temperature", 0.7))
        top_p = float(kwargs.get("top_p", 1.0))
        top_k = int(kwargs.get("top_k", 40))
        repeat_penalty = float(kwargs.get("repeat_penalty", 1.0))
        stop = kwargs.get("stop")

        gen_kwargs: dict[str, object] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
        }
        if stop:
            gen_kwargs["stop"] = stop

        if stream:
            return self._stream_response(messages, gen_kwargs)

        response = self._llm.create_chat_completion(
            messages=messages,
            **gen_kwargs,
        )

        content = response["choices"][0]["message"]["content"]
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "model": self._model_path or "",
        }

    def _stream_response(
        self,
        messages: list[dict],  # type: ignore[type-arg]
        gen_kwargs: dict[str, object],
    ) -> Iterator[dict]:  # type: ignore[type-arg]
        """Stream chat completion response."""
        if not self._llm:
            raise RuntimeError("Model not loaded. Call load() first.")

        stream = self._llm.create_chat_completion(
            messages=messages,
            stream=True,
            **gen_kwargs,
        )

        for chunk in stream:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield {"choices": [{"delta": {"content": content}}]}
