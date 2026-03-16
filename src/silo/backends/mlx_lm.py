"""MLX-LM backend for chat completions."""

from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import Any


class MlxLmBackend:
    """Chat backend wrapping mlx_lm for text generation."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_path: str = ""

    def load(self, model_path: str, config: dict | None = None) -> None:  # type: ignore[type-arg]
        """Load an MLX model into memory."""
        try:
            from mlx_lm import load as mlx_load
        except ImportError as e:
            raise ImportError(
                "mlx-lm is required for MLX chat backend. "
                "Install with: uv pip install 'silo[mlx]'"
            ) from e

        self._model, self._tokenizer = mlx_load(model_path)
        self._model_path = model_path

    def unload(self) -> None:
        """Unload model and free memory."""
        self._model = None
        self._tokenizer = None
        self._model_path = ""
        gc.collect()

    def health(self) -> dict[str, Any]:
        """Return health status."""
        return {
            "status": "ok" if self._model is not None else "unloaded",
            "model": self._model_path,
            "backend": "mlx-lm",
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]] | dict[str, Any]:
        """Generate chat completion using mlx_lm."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            from mlx_lm import generate
        except ImportError as e:
            raise ImportError(
                "mlx-lm is required for MLX chat backend. "
                "Install with: uv pip install 'silo[mlx]'"
            ) from e

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = messages[-1].get("content", "")

        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 1.0)
        top_k = kwargs.get("top_k", 0)
        min_p = kwargs.get("min_p", 0.0)
        repeat_penalty = kwargs.get("repeat_penalty", 1.0)

        sampler = self._make_sampler(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            min_p=float(min_p),
        )

        gen_kwargs: dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "sampler": sampler,
        }
        if float(repeat_penalty) != 1.0:
            gen_kwargs["repetition_penalty"] = float(repeat_penalty)

        if stream:
            return self._stream_response(prompt, gen_kwargs)

        response_text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            **gen_kwargs,
        )
        return {
            "choices": [{"message": {"role": "assistant", "content": response_text}}],
            "model": self._model_path,
        }

    @staticmethod
    def _make_sampler(
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
    ):
        """Build a sampler compatible with the installed mlx_lm version."""
        try:
            from mlx_lm.sample_utils import make_sampler

            return make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
            )
        except ImportError:
            return None

    def _stream_response(
        self, prompt: str, gen_kwargs: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        from mlx_lm import stream_generate

        for chunk in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            **gen_kwargs,
        ):
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            yield {
                "choices": [{"delta": {"content": text}}],
                "model": self._model_path,
            }
