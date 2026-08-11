"""Deterministic semantic input and compatibility fingerprinting."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from semantic_cache.models import ChatCompletionRequest, ChatMessage

_WHITESPACE = re.compile(r"\s+")


def _message_text(content: str | list[Any] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            if item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
            elif "text" in item:
                parts.append(str(item["text"]))
    return "\n".join(parts)


def normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_system_prompt(messages: list[ChatMessage]) -> str:
    """Hash normalized system messages so a prompt change invalidates the cache."""
    chunks = [
        normalize_text(_message_text(message.content)) for message in messages if message.role == "system"
    ]
    return sha256_hex("\n".join(chunk for chunk in chunks if chunk))


def build_semantic_input(messages: list[ChatMessage]) -> str:
    """Build the text that is embedded.

    System prompts are excluded: they are compatibility constraints, not
    semantic query content. User and assistant turns are kept so multi-turn
    context participates in similarity.
    """
    parts: list[str] = []
    for message in messages:
        if message.role == "system":
            continue
        text = normalize_text(_message_text(message.content))
        if text:
            parts.append(f"{message.role}: {text}")
    return "\n".join(parts)


def _canonical_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonical_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    return value


def _norm_float(value: float | None, default: float) -> str:
    resolved = default if value is None else value
    return f"{resolved:.6f}"


def build_fingerprint(request: ChatCompletionRequest, provider: str) -> str:
    """SHA-256 of generation-affecting fields.

    Semantically similar prompts only share a cached response when this
    fingerprint matches. Equivalent values (including omitted defaults)
    produce the same hash.
    """
    payload = {
        "provider": provider,
        "model": request.model,
        "system_prompt_hash": hash_system_prompt(request.messages),
        "temperature": _norm_float(request.temperature, 1.0),
        "top_p": _norm_float(request.top_p, 1.0),
        "max_tokens": request.max_tokens or request.max_completion_tokens or 0,
        "n": request.n or 1,
        "stop": request.stop if request.stop else None,
        "presence_penalty": _norm_float(request.presence_penalty, 0.0),
        "frequency_penalty": _norm_float(request.frequency_penalty, 0.0),
        "seed": request.seed,
        "response_format": _canonical_json(request.response_format),
        "tools": _canonical_json(request.tools),
        "tool_choice": _canonical_json(request.tool_choice),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_hex(canonical)
