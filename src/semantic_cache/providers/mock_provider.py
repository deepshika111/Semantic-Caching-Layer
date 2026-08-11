from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from semantic_cache.models import ChatCompletionRequest


def _last_user_text(request: ChatCompletionRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user" and isinstance(message.content, str):
            return message.content
    return ""


def _response(request: ChatCompletionRequest) -> dict[str, Any]:
    prompt = _last_user_text(request)
    content = f"Mock answer: {prompt}" if prompt else "Mock answer."
    created = int(time.time())
    prompt_tokens = max(len(prompt.split()), 1)
    completion_tokens = max(len(content.split()), 1)
    return {
        "id": f"chatcmpl-mock-{created}",
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class MockProvider:
    """Deterministic local provider for tests and infrastructure benchmarks."""

    name = "mock"

    def __init__(self, delay_seconds: float = 0.01) -> None:
        self._delay_seconds = delay_seconds

    async def complete(self, request: ChatCompletionRequest, api_key: str | None = None) -> dict[str, Any]:
        del api_key
        await asyncio.sleep(self._delay_seconds)
        return _response(request)

    async def stream(self, request: ChatCompletionRequest, api_key: str | None = None) -> AsyncIterator[str]:
        del api_key
        completion = _response(request)
        content = completion["choices"][0]["message"]["content"]
        words = content.split(" ")
        created = completion["created"]
        for index, word in enumerate(words):
            piece = word if index == len(words) - 1 else word + " "
            chunk = {
                "id": completion["id"],
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(self._delay_seconds / max(len(words), 1))
        final = {
            "id": completion["id"],
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
