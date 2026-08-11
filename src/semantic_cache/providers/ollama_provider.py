from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from semantic_cache.config import Settings
from semantic_cache.models import ChatCompletionRequest
from semantic_cache.providers.base import ProviderError


def _messages(request: ChatCompletionRequest) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for message in request.messages:
        text = message.content if isinstance(message.content, str) else ""
        converted.append({"role": message.role, "content": text})
    return converted


def _model_name(model: str) -> str:
    return model.split("/", 1)[1] if model.startswith("ollama/") else model


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _url(self) -> str:
        return self._settings.ollama_base_url.rstrip("/") + "/api/chat"

    async def complete(self, request: ChatCompletionRequest, api_key: str | None = None) -> dict[str, Any]:
        del api_key
        payload = {
            "model": _model_name(request.model),
            "messages": _messages(request),
            "stream": False,
        }
        if request.temperature is not None:
            payload["options"] = {"temperature": request.temperature}
        async with httpx.AsyncClient(timeout=self._settings.provider_timeout_seconds) as client:
            response = await client.post(self._url(), json=payload)
        if response.status_code >= 400:
            raise ProviderError("Ollama provider error", status_code=response.status_code)
        data = response.json()
        message = data.get("message") or {}
        return {
            "id": f"chatcmpl-ollama-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": message.get("role") or "assistant",
                        "content": message.get("content") or "",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count") or 0,
                "completion_tokens": data.get("eval_count") or 0,
                "total_tokens": (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0),
            },
        }

    async def stream(self, request: ChatCompletionRequest, api_key: str | None = None) -> AsyncIterator[str]:
        del api_key
        payload = {
            "model": _model_name(request.model),
            "messages": _messages(request),
            "stream": True,
        }
        created = int(time.time())
        completion_id = f"chatcmpl-ollama-{created}"
        async with httpx.AsyncClient(timeout=self._settings.provider_timeout_seconds) as client:
            async with client.stream("POST", self._url(), json=payload) as response:
                if response.status_code >= 400:
                    raise ProviderError("Ollama provider error", status_code=response.status_code)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = ((event.get("message") or {}).get("content")) or ""
                    done = bool(event.get("done"))
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text} if text else {},
                                "finish_reason": "stop" if done else None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if done:
                        break
        yield "data: [DONE]\n\n"
