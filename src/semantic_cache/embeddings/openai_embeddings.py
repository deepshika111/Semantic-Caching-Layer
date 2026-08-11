from __future__ import annotations
import logging
from openai import APIError, APITimeoutError, AsyncOpenAI, AuthenticationError
from semantic_cache.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class OpenAIEmbeddingService:
    def __init__(self, settings: Settings, api_key: str | None = None) -> None:
        key = api_key or settings.openai_api_key
        if not key:
            raise EmbeddingError("OPENAI_API_KEY is required for embeddings")
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=key,
            base_url=settings.openai_base_url or None,
            timeout=settings.embedding_timeout_seconds,
        )

    async def embed(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            raise EmbeddingError("Cannot embed empty input")
        try:
            response = await self._client.embeddings.create(
                model=self._settings.embedding_model,
                input=cleaned,
                dimensions=self._settings.embedding_dimensions,
                timeout=self._settings.embedding_timeout_seconds,
            )
        except AuthenticationError as exc:
            raise EmbeddingError("Embedding authentication failed") from exc
        except APITimeoutError as exc:
            raise EmbeddingError("Embedding request timed out") from exc
        except APIError as exc:
            raise EmbeddingError("Embedding provider error") from exc
        if not response.data or not response.data[0].embedding:
            raise EmbeddingError("Embedding provider returned an empty vector")
        return list(response.data[0].embedding)

    async def close(self) -> None:
        await self._client.close()