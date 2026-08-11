from __future__ import annotations

from typing import Protocol

from semantic_cache.config import Settings
from semantic_cache.models import CacheDecision, CacheRecord, SearchHit


class CacheRepository(Protocol):
    async def upsert(self, record: CacheRecord) -> None: ...

    async def search(
        self,
        embedding: list[float],
        fingerprint: str,
        tenant: str,
        top_k: int,
        now: float,
    ) -> list[SearchHit]: ...

    async def get(self, cache_id: str) -> CacheRecord | None: ...

    async def increment_hit(self, cache_id: str) -> None: ...

    async def delete_by_model(self, model: str) -> int: ...

    async def delete_by_system_hash(self, system_prompt_hash: str) -> int: ...

    async def delete_by_tag(self, tag: str) -> int: ...

    async def count(self) -> int: ...

    async def ping(self) -> bool: ...

    async def log_decision(self, decision: CacheDecision, max_entries: int) -> None: ...

    async def recent_decisions(self, limit: int = 1000) -> list[CacheDecision]: ...

    async def set_feedback(
        self,
        label: str,
        decision_id: str | None = None,
        cache_id: str | None = None,
    ) -> int: ...

    async def close(self) -> None: ...


async def build_repository(settings: Settings) -> CacheRepository:
    if settings.cache_backend == "memory":
        from semantic_cache.cache.memory_repository import MemoryCacheRepository

        return MemoryCacheRepository()

    from semantic_cache.cache.redis_repository import RedisCacheRepository

    return await RedisCacheRepository.create(settings)
