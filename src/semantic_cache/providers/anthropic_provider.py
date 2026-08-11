from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from semantic_cache.config import Settings
from semantic_cache.models import ChatCompletionRequest
from semantic_cache.providers.base import ProviderError


def _split_messages(request: ChatCompletionRequest) -> tuple[str | None, list[dict[str, str]]]:
    system_parts: list[str] = []
    converted: list[dict[str, str]] = []
    for message in request.messages:
        text = message.content if isinstance(message.content, str) else ""
        if message.role == "system":
            if text:
                system_parts.append(text)
            continue
        role = "assistant" if message.role == "assistant" else "user"
        converted.append({"role": role, "content": text})
    return ("\n".join(system_parts) or None, converted)


def _to_openai(model: str, data: dict[str, Any]) -> dict[str, Any]:
    content = ""
    for block in data.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            content += block.get("text") or ""
    usage = data.get("usage") or {}
    return {
        "id": data.get("id") or f"chatcmpl-anthropic-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop" if data.get("stop_reason") else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens") or 0,
            "completion_tokens": usage.get("output_tokens") or 0,
            "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
        },
    }


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _headers(self, api_key: str | None) -> dict[str, str]:
        key = api_key or self._settings.anthropic_api_key
        if not key:
            raise ProviderError("Anthropic API key is not configured", status_code=401)
        return {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _body(self, request: ChatCompletionRequest) -> dict[str, Any]:
        system, messages = _split_messages(request)
        max_tokens = request.max_tokens or request.max_completion_tokens or 1024
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            body["system"] = system
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        return body

    async def complete(self, request: ChatCompletionRequest, api_key: str | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._settings.provider_timeout_seconds) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=self._headers(api_key),
                json=self._body(request),
            )
        if response.status_code >= 400:
            raise ProviderError("Anthropic provider error", status_code=response.status_code)
        return _to_openai(request.model, response.json())

    async def stream(self, request: ChatCompletionRequest, api_key: str | None = None) -> AsyncIterator[str]:
        body = self._body(request)
        body["stream"] = True
        created = int(time.time())
        completion_id = f"chatcmpl-anthropic-{created}"
        async with httpx.AsyncClient(timeout=self._settings.provider_timeout_seconds) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers=self._headers(api_key),
                json=body,
            ) as response:
                if response.status_code >= 400:
                    raise ProviderError("Anthropic provider error", status_code=response.status_code)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    event_type = event.get("type")
                    if event_type == "content_block_delta":
                        text = (event.get("delta") or {}).get("text") or ""
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif event_type == "message_stop":
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
