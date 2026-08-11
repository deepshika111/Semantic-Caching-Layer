from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from semantic_cache.models import ChatCompletionRequest


class ProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMProvider(Protocol):
    name: str

    async def complete(
        self, request: ChatCompletionRequest, api_key: str | None = None
    ) -> dict[str, Any]: ...

    def stream(self, request: ChatCompletionRequest, api_key: str | None = None) -> AsyncIterator[str]: ...
