from __future__ import annotations

from semantic_cache.cache.similarity import cosine_similarity
from semantic_cache.models import CacheDecision, CacheRecord, SearchHit


class MemoryCacheRepository:
    """In-memory vector store used by tests and CACHE_BACKEND=memory."""

    def __init__(self) -> None:
        self._records: dict[str, CacheRecord] = {}
        self._decisions: list[CacheDecision] = []

    async def upsert(self, record: CacheRecord) -> None:
        self._records[record.cache_id] = record.model_copy(deep=True)

    async def search(
        self,
        embedding: list[float],
        fingerprint: str,
        tenant: str,
        top_k: int,
        now: float,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        expired: list[str] = []
        for cache_id, record in self._records.items():
            if record.expires_at <= now:
                expired.append(cache_id)
                continue
            if record.fingerprint != fingerprint or record.tenant != tenant:
                continue
            similarity = cosine_similarity(embedding, record.embedding)
            hits.append(SearchHit(record=record.model_copy(deep=True), similarity=similarity))
            if similarity >= 0.999:
                break
        for cache_id in expired:
            self._records.pop(cache_id, None)
        hits.sort(key=lambda item: item.similarity, reverse=True)
        return hits[:top_k]

    async def get(self, cache_id: str) -> CacheRecord | None:
        record = self._records.get(cache_id)
        return record.model_copy(deep=True) if record else None

    async def increment_hit(self, cache_id: str) -> None:
        record = self._records.get(cache_id)
        if record:
            record.hit_count += 1

    async def delete_by_model(self, model: str) -> int:
        return self._delete_where(lambda record: record.model == model)

    async def delete_by_system_hash(self, system_prompt_hash: str) -> int:
        return self._delete_where(lambda record: record.system_prompt_hash == system_prompt_hash)

    async def delete_by_tag(self, tag: str) -> int:
        return self._delete_where(lambda record: tag in record.tags)

    def _delete_where(self, predicate) -> int:
        to_delete = [cache_id for cache_id, record in self._records.items() if predicate(record)]
        for cache_id in to_delete:
            del self._records[cache_id]
        return len(to_delete)

    async def count(self) -> int:
        return len(self._records)

    async def ping(self) -> bool:
        return True

    async def log_decision(self, decision: CacheDecision, max_entries: int) -> None:
        self._decisions.append(decision)
        if len(self._decisions) > max_entries:
            self._decisions = self._decisions[-max_entries:]

    async def recent_decisions(self, limit: int = 1000) -> list[CacheDecision]:
        return list(self._decisions[-limit:])

    async def set_feedback(
        self,
        label: str,
        decision_id: str | None = None,
        cache_id: str | None = None,
    ) -> int:
        updated = 0
        for decision in self._decisions:
            if decision_id and decision.decision_id == decision_id:
                decision.feedback = label  # type: ignore[assignment]
                updated += 1
            elif cache_id and decision.cache_id == cache_id:
                decision.feedback = label  # type: ignore[assignment]
                updated += 1
        return updated

    async def close(self) -> None:
        return None

    def evict_oldest(self, keep: int) -> int:
        if len(self._records) <= keep:
            return 0
        ordered = sorted(self._records.values(), key=lambda record: record.created_at)
        removed = 0
        while len(self._records) > keep and ordered:
            record = ordered.pop(0)
            if record.cache_id in self._records:
                del self._records[record.cache_id]
                removed += 1
        return removed
