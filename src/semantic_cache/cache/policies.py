"""Deterministic cacheability, TTL, and per-type similarity thresholds."""

from __future__ import annotations

import re

from semantic_cache.config import Settings
from semantic_cache.models import CachePolicyResult, ChatCompletionRequest, ChatMessage

TEMPORAL_TERMS = (
    "today",
    "right now",
    "latest",
    "current",
    "currently",
    "this week",
    "breaking",
    "recent",
    "live",
    "weather",
    "stock price",
    "this morning",
    "tonight",
    "yesterday",
    "as of",
)

CLASSIFICATION_TERMS = (
    "classify",
    "classification",
    "category",
    "label this",
    "sentiment",
    "yes or no",
    "true or false",
    "choose one",
    "which category",
    "is this",
)

CREATIVE_TERMS = (
    "write a poem",
    "write a story",
    "short story",
    "haiku",
    "lyrics",
    "imagine",
    "make up",
    "roleplay",
    "role-play",
    "role play",
    "creative writing",
)

FACTUAL_TERMS = (
    "what is",
    "what are",
    "who is",
    "who was",
    "explain",
    "define",
    "how does",
    "how do",
    "why does",
    "why do",
    "summarize",
    "overview",
)

_WORD_BOUNDARY_TERMS = {"live", "today", "latest", "current", "currently", "recent", "weather"}


def _user_text(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.role != "user":
            continue
        content = message.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
    return " ".join(parts).lower()


def _contains_term(text: str, term: str) -> bool:
    if term in _WORD_BOUNDARY_TERMS or " " not in term:
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term in text


def classify_request(request: ChatCompletionRequest) -> str:
    """Heuristic request type. Priority: temporal > creative > classification > factual."""
    text = _user_text(request.messages)
    temperature = 1.0 if request.temperature is None else request.temperature

    if any(_contains_term(text, term) for term in TEMPORAL_TERMS):
        return "temporal"
    if temperature >= 1.2 or any(_contains_term(text, term) for term in CREATIVE_TERMS):
        return "creative"
    if any(_contains_term(text, term) for term in CLASSIFICATION_TERMS):
        return "classification"
    if any(_contains_term(text, term) for term in FACTUAL_TERMS):
        return "factual"
    return "other"


def threshold_for(request_type: str, settings: Settings) -> float:
    mapping = {
        "classification": settings.classification_threshold,
        "factual": settings.factual_threshold,
        "creative": settings.creative_threshold,
        "temporal": settings.semantic_cache_threshold,
        "other": settings.semantic_cache_threshold,
    }
    return mapping.get(request_type, settings.semantic_cache_threshold)


def evaluate_policy(request: ChatCompletionRequest, settings: Settings) -> CachePolicyResult:
    request_type = classify_request(request)
    tags = [request_type, f"model:{request.model}"]

    if not settings.cache_enabled:
        return CachePolicyResult(
            cacheable=False,
            reason="cache_disabled",
            ttl_seconds=0,
            threshold=settings.semantic_cache_threshold,
            request_type=request_type,
            tags=tags,
        )

    if not request.messages:
        return CachePolicyResult(
            cacheable=False,
            reason="empty_messages",
            ttl_seconds=0,
            threshold=settings.semantic_cache_threshold,
            request_type=request_type,
            tags=tags,
        )

    if request.tools:
        return CachePolicyResult(
            cacheable=False,
            reason="tool_calls_not_cached",
            ttl_seconds=0,
            threshold=settings.semantic_cache_threshold,
            request_type=request_type,
            tags=tags,
        )

    if request.n is not None and request.n > 1:
        return CachePolicyResult(
            cacheable=False,
            reason="multiple_choices_not_cached",
            ttl_seconds=0,
            threshold=settings.semantic_cache_threshold,
            request_type=request_type,
            tags=tags,
        )

    if request_type == "creative" and not settings.creative_cache_enabled:
        return CachePolicyResult(
            cacheable=False,
            reason="creative_disabled",
            ttl_seconds=0,
            threshold=settings.creative_threshold,
            request_type=request_type,
            tags=tags,
        )

    if request_type == "temporal" and settings.temporal_cache_mode == "no_cache":
        return CachePolicyResult(
            cacheable=False,
            reason="temporal_no_cache",
            ttl_seconds=0,
            threshold=settings.semantic_cache_threshold,
            request_type=request_type,
            tags=tags,
        )

    ttl = settings.default_cache_ttl_seconds
    if request_type == "temporal":
        ttl = settings.temporal_cache_ttl_seconds

    if ttl <= 0:
        return CachePolicyResult(
            cacheable=False,
            reason="zero_ttl",
            ttl_seconds=0,
            threshold=threshold_for(request_type, settings),
            request_type=request_type,
            tags=tags,
        )

    return CachePolicyResult(
        cacheable=True,
        reason="cacheable",
        ttl_seconds=ttl,
        threshold=threshold_for(request_type, settings),
        request_type=request_type,
        tags=tags,
    )
