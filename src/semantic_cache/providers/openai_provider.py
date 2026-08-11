from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, AuthenticationError, RateLimitError

from semantic_cache.config import Settings
from semantic_cache.models import ChatCompletionRequest
from semantic_cache.providers.base import ProviderError


def _payload(request: ChatCompletionRequest) -> dict[str, Any]:
    data = request.model_dump(exclude_none=True)
    data.pop("stream", None)
    return data


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _client(self, api_key: str | None) -> AsyncOpenAI:
        key = api_key or self._settings.openai_api_key
        if not key:
            raise ProviderError("OpenAI API key is not configured", status_code=401)
        return AsyncOpenAI(
            api_key=key,
            base_url=self._settings.openai_base_url or None,
            timeout=self._settings.provider_timeout_seconds,
        )

    async def complete(self, request: ChatCompletionRequest, api_key: str | None = None) -> dict[str, Any]:
        client = self._client(api_key)
        try:
            response = await client.chat.completions.create(**_payload(request), stream=False)
            return response.model_dump(mode="json")
        except AuthenticationError as exc:
            raise ProviderError("OpenAI authentication failed", status_code=401) from exc
        except RateLimitError as exc:
            raise ProviderError("OpenAI rate limit exceeded", status_code=429) from exc
        except APITimeoutError as exc:
            raise ProviderError("OpenAI request timed out", status_code=504) from exc
        except APIError as exc:
            status = getattr(exc, "status_code", 502) or 502
            raise ProviderError("OpenAI provider error", status_code=int(status)) from exc
        finally:
            await client.close()

    async def stream(self, request: ChatCompletionRequest, api_key: str | None = None) -> AsyncIterator[str]:
        client = self._client(api_key)
        try:
            stream = await client.chat.completions.create(**_payload(request), stream=True)
            async for chunk in stream:
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        except AuthenticationError as exc:
            raise ProviderError("OpenAI authentication failed", status_code=401) from exc
        except RateLimitError as exc:
            raise ProviderError("OpenAI rate limit exceeded", status_code=429) from exc
        except APITimeoutError as exc:
            raise ProviderError("OpenAI request timed out", status_code=504) from exc
        except APIError as exc:
            status = getattr(exc, "status_code", 502) or 502
            raise ProviderError("OpenAI provider error", status_code=int(status)) from exc
        finally:
            await client.close()