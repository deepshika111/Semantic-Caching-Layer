from __future__ import annotations

import time
import uuid
from typing import Any

from semantic_cache.cache.key_builder import sha256_hex
from semantic_cache.cache.repository import CacheRepository
from semantic_cache.config import Settings
from semantic_cache.models import CacheDecision, CachePolicyResult, CacheRecord, SearchHit


class SemanticCache:
    def __init__(self, repository: CacheRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    async def lookup(
        self,
        embedding: list[float],
        fingerprint: str,
        tenant: str,
        threshold: float,
    ) -> SearchHit | None:
        hits = await self._repository.search(
            embedding=embedding,
            fingerprint=fingerprint,
            tenant=tenant,
            top_k=self._settings.cache_search_top_k,
            now=time.time(),
        )
        if not hits:
            return None
        return hits[0]

    def classify_outcome(self, similarity: float | None, threshold: float) -> str:
        if similarity is None:
            return "miss"
        if similarity >= threshold:
            return "hit"
        floor = max(0.0, threshold - self._settings.near_miss_delta)
        if similarity >= floor:
            return "near_miss"
        return "miss"

    async def store(
        self,
        *,
        tenant: str,
        semantic_input: str,
        embedding: list[float],
        response: dict[str, Any],
        provider: str,
        model: str,
        fingerprint: str,
        system_prompt_hash: str,
        policy: CachePolicyResult,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        finish_reason: str | None = None,
    ) -> CacheRecord:
        now = time.time()
        record = CacheRecord(
            cache_id=str(uuid.uuid4()),
            tenant=tenant,
            semantic_input=semantic_input if self._settings.prompt_retention_enabled else "",
            semantic_input_hash=sha256_hex(semantic_input),
            embedding=embedding,
            response=response,
            provider=provider,
            model=model,
            fingerprint=fingerprint,
            system_prompt_hash=system_prompt_hash,
            request_type=policy.request_type,
            tags=policy.tags,
            created_at=now,
            expires_at=now + policy.ttl_seconds,
            ttl_seconds=policy.ttl_seconds,
            hit_count=0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
        await self._evict_if_needed()
        await self._repository.upsert(record)
        return record

    async def _evict_if_needed(self) -> int:
        count = await self._repository.count()
        if count < self._settings.max_cache_entries:
            return 0
        evict_fn = getattr(self._repository, "evict_oldest", None)
        if evict_fn is None:
            return 0
        overflow = count - self._settings.max_cache_entries + 1
        keep = max(self._settings.max_cache_entries - overflow, 0)
        return int(evict_fn(keep))

    async def record_decision(
        self,
        *,
        tenant: str,
        provider: str,
        model: str,
        request_type: str,
        similarity: float | None,
        threshold: float,
        outcome: str,
        cache_id: str | None,
        fingerprint: str | None,
    ) -> CacheDecision:
        decision = CacheDecision(
            decision_id=str(uuid.uuid4()),
            timestamp=time.time(),
            tenant=tenant,
            provider=provider,
            model=model,
            request_type=request_type,
            similarity=similarity,
            threshold=threshold,
            outcome=outcome,  # type: ignore[arg-type]
            cache_id=cache_id,
            fingerprint=fingerprint,
        )
        await self._repository.log_decision(decision, self._settings.decision_log_max_entries)
        return decision

    async def increment_hit(self, cache_id: str) -> None:
        await self._repository.increment_hit(cache_id)

    async def invalidate_model(self, model: str) -> int:
        return await self._repository.delete_by_model(model)

    async def invalidate_system(self, system_hash: str) -> int:
        return await self._repository.delete_by_system_hash(system_hash)

    async def invalidate_tag(self, tag: str) -> int:
        return await self._repository.delete_by_tag(tag)

    async def entry_count(self) -> int:
        return await self._repository.count()

    async def recent_decisions(self, limit: int = 1000) -> list[CacheDecision]:
        return await self._repository.recent_decisions(limit)

    async def set_feedback(
        self,
        label: str,
        decision_id: str | None = None,
        cache_id: str | None = None,
    ) -> int:
        return await self._repository.set_feedback(label, decision_id, cache_id)
