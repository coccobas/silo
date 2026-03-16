"""Chat completion routes — OpenAI-compatible."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from silo.server.errors import MODEL_NOT_FOUND, openai_error_response
from silo.server.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    DeltaContent,
    StreamChoice,
    Usage,
)

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request_body: ChatCompletionRequest, request: Request):  # type: ignore[no-untyped-def]
    """OpenAI-compatible chat completion endpoint."""
    backend = request.app.state.backend
    model_name = request.app.state.model_name

    if request_body.model != model_name:
        return openai_error_response(
            404,
            f"Model '{request_body.model}' not found. This server serves '{model_name}'.",
            MODEL_NOT_FOUND,
        )

    messages = [{"role": m.role, "content": m.content} for m in request_body.messages]

    if request_body.stream:
        return _stream_response(backend, messages, request_body, model_name)

    result = backend.chat(
        messages,
        stream=False,
        max_tokens=request_body.max_tokens,
        temperature=request_body.temperature,
        top_p=request_body.top_p,
        top_k=request_body.top_k,
        min_p=request_body.min_p,
        repeat_penalty=request_body.repeat_penalty,
        stop=request_body.stop,
    )
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    prompt_tokens = sum(len(m.content.split()) for m in request_body.messages)
    completion_tokens = len(content.split())

    return ChatCompletionResponse(
        model=model_name,
        choices=[Choice(message=ChoiceMessage(content=content))],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _stream_response(
    backend, messages: list[dict], request_body: ChatCompletionRequest, model_name: str  # type: ignore[type-arg]
) -> EventSourceResponse:
    """Create an SSE streaming response."""
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async def event_generator():  # type: ignore[no-untyped-def]
        result = backend.chat(
            messages,
            stream=True,
            max_tokens=request_body.max_tokens,
            temperature=request_body.temperature,
            top_p=request_body.top_p,
            top_k=request_body.top_k,
            min_p=request_body.min_p,
            repeat_penalty=request_body.repeat_penalty,
            stop=request_body.stop,
        )

        for chunk_data in result:
            content = (
                chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            )
            chunk = ChatCompletionChunk(
                id=stream_id,
                model=model_name,
                choices=[StreamChoice(delta=DeltaContent(content=content))],
            )
            yield {"data": chunk.model_dump_json()}

        # Final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id=stream_id,
            model=model_name,
            choices=[
                StreamChoice(
                    delta=DeltaContent(content=""), finish_reason="stop"
                )
            ],
        )
        yield {"data": final_chunk.model_dump_json()}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())
