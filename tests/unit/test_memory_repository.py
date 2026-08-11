import time

import pytest

from semantic_cache.cache.memory_repository import MemoryCacheRepository
from semantic_cache.models import CacheRecord


def _record(**overrides) -> CacheRecord:
    now = time.time()
    data = {
        "cache_id": "a",
        "tenant": "default",
        "semantic_input": "user: hello",
        "semantic_input_hash": "abc",
        "embedding": [1.0, 0.0],
        "response": {"ok": True},
        "provider": "mock",
        "model": "gpt-4o-mini",
        "fingerprint": "fp1",
        "system_prompt_hash": "sys",
        "request_type": "factual",
        "tags": ["factual"],
        "created_at": now,
        "expires_at": now + 60,
        "ttl_seconds": 60,
    }
    data.update(overrides)
    return CacheRecord(**data)


@pytest.mark.asyncio
async def test_search_filters_fingerprint_and_expiry() -> None:
    repo = MemoryCacheRepository()
    await repo.upsert(_record())
    await repo.upsert(_record(cache_id="b", fingerprint="fp2", embedding=[0.0, 1.0]))
    await repo.upsert(_record(cache_id="c", expires_at=time.time() - 1, embedding=[1.0, 0.0]))
    hits = await repo.search([1.0, 0.0], "fp1", "default", top_k=5, now=time.time())
    assert len(hits) == 1
    assert hits[0].record.cache_id == "a"
    assert hits[0].similarity > 0.99


@pytest.mark.asyncio
async def test_invalidation_by_tag_and_model() -> None:
    repo = MemoryCacheRepository()
    await repo.upsert(_record(cache_id="a", tags=["factual", "model:gpt-4o-mini"]))
    await repo.upsert(_record(cache_id="b", model="gpt-4o", tags=["factual"]))
    assert await repo.delete_by_tag("factual") == 2
    await repo.upsert(_record(cache_id="c"))
    assert await repo.delete_by_model("gpt-4o-mini") == 1
