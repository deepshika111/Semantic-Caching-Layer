from __future__ import annotations

import json
import logging
import struct
from typing import Any

from redis.asyncio import Redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from semantic_cache.config import Settings
from semantic_cache.models import CacheDecision, CacheRecord, SearchHit

logger = logging.getLogger(__name__)

DECISIONS_KEY = "scache:decisions"


def _pack_embedding(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_embedding(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    return value


def _escape_tag(value: str) -> str:
    special = set(r'\,.<>{}[]"\':;!@#$%^&*()-+=~| ')
    return "".join(f"\\{char}" if char in special else char for char in value)


def _record_mapping(record: CacheRecord, retain_prompt: bool) -> dict[str, Any]:
    return {
        "cache_id": record.cache_id,
        "tenant": record.tenant,
        "fingerprint": record.fingerprint,
        "provider": record.provider,
        "model": record.model,
        "system_prompt_hash": record.system_prompt_hash,
        "request_type": record.request_type,
        "tags": ",".join(record.tags),
        "semantic_input": record.semantic_input if retain_prompt else "",
        "semantic_input_hash": record.semantic_input_hash,
        "response_json": json.dumps(record.response, separators=(",", ":")),
        "created_at": record.created_at,
        "expires_at": record.expires_at,
        "ttl_seconds": record.ttl_seconds,
        "hit_count": record.hit_count,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "finish_reason": record.finish_reason or "",
        "embedding": _pack_embedding(record.embedding),
    }


def _doc_to_record(doc: dict[str, Any]) -> CacheRecord:
    embedding = doc.get("embedding")
    if isinstance(embedding, bytes):
        embedding = _unpack_embedding(embedding)
    elif not embedding:
        embedding = []
    tags_raw = _decode(doc.get("tags") or "")
    response_raw = _decode(doc.get("response_json") or "{}")
    return CacheRecord(
        cache_id=_decode(doc.get("cache_id") or ""),
        tenant=_decode(doc.get("tenant") or "default"),
        semantic_input=_decode(doc.get("semantic_input") or ""),
        semantic_input_hash=_decode(doc.get("semantic_input_hash") or ""),
        embedding=list(embedding),
        response=json.loads(response_raw),
        provider=_decode(doc.get("provider") or ""),
        model=_decode(doc.get("model") or ""),
        fingerprint=_decode(doc.get("fingerprint") or ""),
        system_prompt_hash=_decode(doc.get("system_prompt_hash") or ""),
        request_type=_decode(doc.get("request_type") or "other"),
        tags=[tag for tag in tags_raw.split(",") if tag],
        created_at=float(_decode(doc.get("created_at") or 0)),
        expires_at=float(_decode(doc.get("expires_at") or 0)),
        ttl_seconds=int(float(_decode(doc.get("ttl_seconds") or 0))),
        hit_count=int(float(_decode(doc.get("hit_count") or 0))),
        prompt_tokens=int(float(_decode(doc.get("prompt_tokens") or 0))),
        completion_tokens=int(float(_decode(doc.get("completion_tokens") or 0))),
        finish_reason=_decode(doc.get("finish_reason") or "") or None,
    )


class RedisCacheRepository:
    """Redis Stack vector index (RediSearch KNN) over HASH documents."""

    def __init__(self, settings: Settings, client: Redis) -> None:
        self._settings = settings
        self._client = client

    @classmethod
    async def create(cls, settings: Settings) -> RedisCacheRepository:
        client = Redis.from_url(settings.redis_url, decode_responses=False)
        repo = cls(settings, client)
        await repo._ensure_index()
        return repo

    async def _ensure_index(self) -> None:
        fields = [
            TagField("cache_id"),
            TagField("tenant"),
            TagField("fingerprint"),
            TagField("provider"),
            TagField("model"),
            TagField("system_prompt_hash"),
            TagField("request_type"),
            TagField("tags", separator=","),
            TextField("semantic_input"),
            TagField("semantic_input_hash"),
            TextField("response_json"),
            NumericField("created_at"),
            NumericField("expires_at"),
            NumericField("ttl_seconds"),
            NumericField("hit_count"),
            NumericField("prompt_tokens"),
            NumericField("completion_tokens"),
            TagField("finish_reason"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self._settings.embedding_dimensions,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        ]
        definition = IndexDefinition(
            prefix=[f"{self._settings.redis_key_prefix}:entry:"],
            index_type=IndexType.HASH,
        )
        try:
            await self._client.ft(self._settings.redis_index_name).create_index(
                fields=fields,
                definition=definition,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "already exists" not in message and "index already" not in message:
                raise

    def _key(self, cache_id: str) -> str:
        return f"{self._settings.redis_key_prefix}:entry:{cache_id}"

    async def upsert(self, record: CacheRecord) -> None:
        mapping = _record_mapping(record, self._settings.prompt_retention_enabled)
        await self._client.hset(self._key(record.cache_id), mapping=mapping)
        await self._client.expire(self._key(record.cache_id), max(record.ttl_seconds, 1))

    async def search(
        self,
        embedding: list[float],
        fingerprint: str,
        tenant: str,
        top_k: int,
        now: float,
    ) -> list[SearchHit]:
        filter_query = (
            f"(@fingerprint:{{{_escape_tag(fingerprint)}}} "
            f"@tenant:{{{_escape_tag(tenant)}}} "
            f"@expires_at:[{int(now)} +inf])"
        )
        knn = f"=>[KNN {top_k} @embedding $vec AS vector_distance]"
        query = (
            Query(filter_query + knn)
            .return_fields(
                "cache_id",
                "tenant",
                "fingerprint",
                "provider",
                "model",
                "system_prompt_hash",
                "request_type",
                "tags",
                "semantic_input",
                "semantic_input_hash",
                "response_json",
                "created_at",
                "expires_at",
                "ttl_seconds",
                "hit_count",
                "prompt_tokens",
                "completion_tokens",
                "finish_reason",
                "vector_distance",
            )
            .sort_by("vector_distance")
            .paging(0, top_k)
            .dialect(2)
        )
        try:
            result = await self._client.ft(self._settings.redis_index_name).search(
                query,
                query_params={"vec": _pack_embedding(embedding)},
            )
        except Exception:
            logger.exception("Redis vector search failed")
            return []

        hits: list[SearchHit] = []
        for doc in result.docs:
            data = dict(doc.__dict__)
            data.pop("payload", None)
            distance = float(data.get("vector_distance") or 1.0)
            similarity = max(0.0, min(1.0, 1.0 - distance))
            try:
                record = _doc_to_record(data)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if record.cache_id:
                hits.append(SearchHit(record=record, similarity=similarity))
        return hits

    async def get(self, cache_id: str) -> CacheRecord | None:
        raw = await self._client.hgetall(self._key(cache_id))
        if not raw:
            return None
        decoded: dict[str, Any] = {}
        for key, value in raw.items():
            field = key.decode() if isinstance(key, bytes) else str(key)
            if field == "embedding":
                decoded[field] = value
            else:
                decoded[field] = value.decode() if isinstance(value, bytes) else value
        return _doc_to_record(decoded)

    async def increment_hit(self, cache_id: str) -> None:
        await self._client.hincrby(self._key(cache_id), "hit_count", 1)

    async def _scan_delete(self, field: str, expected: str) -> int:
        deleted = 0
        async for key in self._client.scan_iter(match=f"{self._settings.redis_key_prefix}:entry:*"):
            value = await self._client.hget(key, field)
            if value is None:
                continue
            text = value.decode() if isinstance(value, bytes) else str(value)
            if field == "tags":
                tags = {tag.strip() for tag in text.split(",") if tag.strip()}
                if expected not in tags:
                    continue
            elif text != expected:
                continue
            await self._client.delete(key)
            deleted += 1
        return deleted

    async def delete_by_model(self, model: str) -> int:
        return await self._scan_delete("model", model)

    async def delete_by_system_hash(self, system_prompt_hash: str) -> int:
        return await self._scan_delete("system_prompt_hash", system_prompt_hash)

    async def delete_by_tag(self, tag: str) -> int:
        return await self._scan_delete("tags", tag)

    async def count(self) -> int:
        n = 0
        async for _ in self._client.scan_iter(match=f"{self._settings.redis_key_prefix}:entry:*"):
            n += 1
        return n

    async def ping(self) -> bool:
        return bool(await self._client.ping())

    async def log_decision(self, decision: CacheDecision, max_entries: int) -> None:
        await self._client.lpush(DECISIONS_KEY, decision.model_dump_json())
        await self._client.ltrim(DECISIONS_KEY, 0, max_entries - 1)

    async def recent_decisions(self, limit: int = 1000) -> list[CacheDecision]:
        raw = await self._client.lrange(DECISIONS_KEY, 0, limit - 1)
        decisions: list[CacheDecision] = []
        for item in raw:
            text = item.decode() if isinstance(item, bytes) else item
            try:
                decisions.append(CacheDecision.model_validate_json(text))
            except (ValueError, json.JSONDecodeError):
                continue
        decisions.reverse()
        return decisions

    async def set_feedback(
        self,
        label: str,
        decision_id: str | None = None,
        cache_id: str | None = None,
    ) -> int:
        raw = await self._client.lrange(DECISIONS_KEY, 0, -1)
        updated = 0
        new_items: list[str] = []
        for item in raw:
            text = item.decode() if isinstance(item, bytes) else item
            try:
                decision = CacheDecision.model_validate_json(text)
            except (ValueError, json.JSONDecodeError):
                new_items.append(text)
                continue
            matched = (decision_id and decision.decision_id == decision_id) or (
                cache_id and decision.cache_id == cache_id
            )
            if matched:
                decision.feedback = label  # type: ignore[assignment]
                updated += 1
            new_items.append(decision.model_dump_json())
        if updated:
            pipe = self._client.pipeline()
            pipe.delete(DECISIONS_KEY)
            if new_items:
                pipe.rpush(DECISIONS_KEY, *new_items)
            await pipe.execute()
        return updated

    async def close(self) -> None:
        await self._client.aclose()
