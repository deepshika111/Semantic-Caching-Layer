from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from semantic_cache.cache.key_builder import build_fingerprint, build_semantic_input, hash_system_prompt
from semantic_cache.cache.policies import evaluate_policy
from semantic_cache.cache.semantic_cache import SemanticCache
from semantic_cache.config import Settings
from semantic_cache.embeddings.base import EmbeddingService
from semantic_cache.models import ChatCompletionRequest
from semantic_cache.monitoring.cost import estimate_cost_usd
from semantic_cache.monitoring.metrics import Metrics
from semantic_cache.providers.base import LLMProvider
from semantic_cache.providers.router import ProviderRouter
from semantic_cache.services.streaming import (
    assemble_openai_stream,
    completion_to_sse,
    is_successful_completion,
)

logger = logging.getLogger(__name__)


@dataclass
class ProxyResult:
    body: dict[str, Any] | None
    stream: AsyncIterator[str] | None
    headers: dict[str, str]
    status_code: int = 200


class ProxyService:
    def __init__(
        self,
        settings: Settings,
        cache: SemanticCache,
        embeddings: EmbeddingService,
        router: ProviderRouter,
        metrics: Metrics,
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._embeddings = embeddings
        self._router = router
        self._metrics = metrics

    async def handle(
        self,
        request: ChatCompletionRequest,
        *,
        tenant: str,
        bypass: bool,
        provider_override: str | None,
        api_key: str | None,
    ) -> ProxyResult:
        started = time.perf_counter()
        provider_name, provider = self._router.get(request.model, provider_override)
        policy = evaluate_policy(request, self._settings)
        labels = {
            "provider": provider_name,
            "model": request.model,
            "request_type": policy.request_type,
        }
        self._metrics.requests_total.labels(**labels).inc()

        if bypass:
            return await self._forward(
                request,
                provider_name,
                provider,
                api_key,
                policy.request_type,
                started,
                outcome="bypass",
                extra_headers={"X-Semantic-Cache": "BYPASS"},
            )

        if not policy.cacheable:
            return await self._forward(
                request,
                provider_name,
                provider,
                api_key,
                policy.request_type,
                started,
                outcome="skip",
                extra_headers={
                    "X-Semantic-Cache": "SKIP",
                    "X-Semantic-Cache-Reason": policy.reason,
                },
            )

        semantic_input = build_semantic_input(request.messages)
        if not semantic_input:
            return await self._forward(
                request,
                provider_name,
                provider,
                api_key,
                policy.request_type,
                started,
                outcome="skip",
                extra_headers={
                    "X-Semantic-Cache": "SKIP",
                    "X-Semantic-Cache-Reason": "empty_semantic_input",
                },
            )

        fingerprint = build_fingerprint(request, provider_name)
        system_hash = hash_system_prompt(request.messages)
        embedding = await self._embeddings.embed(semantic_input)
        best = await self._cache.lookup(embedding, fingerprint, tenant, policy.threshold)
        similarity = best.similarity if best else None
        outcome = self._cache.classify_outcome(similarity, policy.threshold)

        if similarity is not None:
            self._metrics.similarity_score.labels(request_type=policy.request_type).observe(similarity)

        headers = self._cache_headers(outcome, similarity, policy.threshold, policy.request_type)

        if outcome == "hit" and best is not None:
            await self._cache.increment_hit(best.record.cache_id)
            self._metrics.hits_total.labels(**labels).inc()
            saved = estimate_cost_usd(
                best.record.model,
                best.record.prompt_tokens,
                best.record.completion_tokens,
            )
            self._metrics.estimated_cost_saved_usd_total.labels(
                provider=provider_name, model=request.model
            ).inc(saved)
            elapsed = time.perf_counter() - started
            self._metrics.hit_latency_seconds.observe(elapsed)
            self._metrics.request_latency_seconds.labels(outcome="hit").observe(elapsed)
            await self._cache.record_decision(
                tenant=tenant,
                provider=provider_name,
                model=request.model,
                request_type=policy.request_type,
                similarity=similarity,
                threshold=policy.threshold,
                outcome="hit",
                cache_id=best.record.cache_id,
                fingerprint=fingerprint,
            )
            if request.stream:
                return ProxyResult(
                    body=None, stream=_as_stream(completion_to_sse(best.record.response)), headers=headers
                )
            return ProxyResult(body=best.record.response, stream=None, headers=headers)

        if outcome == "near_miss":
            self._metrics.near_misses_total.labels(**labels).inc()

        self._metrics.misses_total.labels(**labels).inc()
        await self._cache.record_decision(
            tenant=tenant,
            provider=provider_name,
            model=request.model,
            request_type=policy.request_type,
            similarity=similarity,
            threshold=policy.threshold,
            outcome=outcome if outcome == "near_miss" else "miss",
            cache_id=best.record.cache_id if best else None,
            fingerprint=fingerprint,
        )

        if request.stream:
            return await self._stream_and_maybe_cache(
                request,
                provider_name,
                provider,
                api_key,
                tenant,
                semantic_input,
                embedding,
                fingerprint,
                system_hash,
                policy,
                headers,
                started,
            )

        return await self._complete_and_maybe_cache(
            request,
            provider_name,
            provider,
            api_key,
            tenant,
            semantic_input,
            embedding,
            fingerprint,
            system_hash,
            policy,
            headers,
            started,
        )

    def _cache_headers(
        self,
        outcome: str,
        similarity: float | None,
        threshold: float,
        request_type: str,
    ) -> dict[str, str]:
        status = {"hit": "HIT", "near_miss": "MISS", "miss": "MISS"}.get(outcome, outcome.upper())
        headers = {
            "X-Semantic-Cache": status,
            "X-Semantic-Cache-Type": request_type,
        }
        if self._settings.debug_cache_headers:
            if similarity is not None:
                headers["X-Semantic-Cache-Similarity"] = f"{similarity:.6f}"
            headers["X-Semantic-Cache-Threshold"] = f"{threshold:.6f}"
        return headers

    async def _complete_and_maybe_cache(
        self,
        request: ChatCompletionRequest,
        provider_name: str,
        provider: LLMProvider,
        api_key: str | None,
        tenant: str,
        semantic_input: str,
        embedding: list[float],
        fingerprint: str,
        system_hash: str,
        policy,
        headers: dict[str, str],
        started: float,
    ) -> ProxyResult:
        provider_started = time.perf_counter()
        self._metrics.provider_requests_total.labels(provider=provider_name, model=request.model).inc()
        response = await provider.complete(request, api_key)
        self._metrics.provider_latency_seconds.labels(provider=provider_name).observe(
            time.perf_counter() - provider_started
        )
        if is_successful_completion(response):
            usage = response.get("usage") or {}
            await self._cache.store(
                tenant=tenant,
                semantic_input=semantic_input,
                embedding=embedding,
                response=response,
                provider=provider_name,
                model=request.model,
                fingerprint=fingerprint,
                system_prompt_hash=system_hash,
                policy=policy,
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                finish_reason=(response.get("choices") or [{}])[0].get("finish_reason"),
            )
            try:
                self._metrics.entries.set(await self._cache.entry_count())
            except Exception:
                logger.debug("Failed to refresh cache entry gauge")
        elapsed = time.perf_counter() - started
        self._metrics.request_latency_seconds.labels(outcome="miss").observe(elapsed)
        return ProxyResult(body=response, stream=None, headers=headers)

    async def _stream_and_maybe_cache(
        self,
        request: ChatCompletionRequest,
        provider_name: str,
        provider: LLMProvider,
        api_key: str | None,
        tenant: str,
        semantic_input: str,
        embedding: list[float],
        fingerprint: str,
        system_hash: str,
        policy,
        headers: dict[str, str],
        started: float,
    ) -> ProxyResult:
        self._metrics.provider_requests_total.labels(provider=provider_name, model=request.model).inc()

        async def generator() -> AsyncIterator[str]:
            buffer: list[str] = []
            completed = False
            provider_started = time.perf_counter()
            try:
                async for chunk in provider.stream(request, api_key):
                    buffer.append(chunk)
                    yield chunk
                completed = True
            finally:
                self._metrics.provider_latency_seconds.labels(provider=provider_name).observe(
                    time.perf_counter() - provider_started
                )
                self._metrics.request_latency_seconds.labels(outcome="miss").observe(
                    time.perf_counter() - started
                )
            if not completed:
                return
            assembled = assemble_openai_stream(buffer)
            if assembled is None or not is_successful_completion(assembled):
                return
            usage = assembled.get("usage") or {}
            await self._cache.store(
                tenant=tenant,
                semantic_input=semantic_input,
                embedding=embedding,
                response=assembled,
                provider=provider_name,
                model=request.model,
                fingerprint=fingerprint,
                system_prompt_hash=system_hash,
                policy=policy,
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                finish_reason=(assembled.get("choices") or [{}])[0].get("finish_reason"),
            )

        return ProxyResult(body=None, stream=generator(), headers=headers)

    async def _forward(
        self,
        request: ChatCompletionRequest,
        provider_name: str,
        provider: LLMProvider,
        api_key: str | None,
        request_type: str,
        started: float,
        outcome: str,
        extra_headers: dict[str, str],
    ) -> ProxyResult:
        extra_headers.setdefault("X-Semantic-Cache-Type", request_type)
        self._metrics.provider_requests_total.labels(provider=provider_name, model=request.model).inc()
        if request.stream:

            async def generator() -> AsyncIterator[str]:
                provider_started = time.perf_counter()
                try:
                    async for chunk in provider.stream(request, api_key):
                        yield chunk
                finally:
                    self._metrics.provider_latency_seconds.labels(provider=provider_name).observe(
                        time.perf_counter() - provider_started
                    )
                    self._metrics.request_latency_seconds.labels(outcome=outcome).observe(
                        time.perf_counter() - started
                    )

            return ProxyResult(body=None, stream=generator(), headers=extra_headers)

        provider_started = time.perf_counter()
        response = await provider.complete(request, api_key)
        self._metrics.provider_latency_seconds.labels(provider=provider_name).observe(
            time.perf_counter() - provider_started
        )
        self._metrics.request_latency_seconds.labels(outcome=outcome).observe(time.perf_counter() - started)
        return ProxyResult(body=response, stream=None, headers=extra_headers)


async def _as_stream(chunks) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


def extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "Bearer "
    if authorization.startswith(prefix):
        token = authorization[len(prefix) :].strip()
        return token or None
    return None
