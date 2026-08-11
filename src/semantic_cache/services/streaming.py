from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any


def assemble_openai_stream(chunks: list[str]) -> dict[str, Any] | None:
    """Rebuild a chat.completion object from SSE chunks.

    Returns None when the stream did not complete successfully.
    """
    completion_id = None
    model = None
    created = int(time.time())
    content_parts: list[str] = []
    finish_reason = None
    usage = None
    saw_done = False

    for raw in chunks:
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                saw_done = True
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            completion_id = obj.get("id") or completion_id
            model = obj.get("model") or model
            created = obj.get("created") or created
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    if not saw_done:
        return None
    if finish_reason is None:
        finish_reason = "stop"
    if not content_parts and finish_reason not in {"stop", "length"}:
        return None

    return {
        "id": completion_id or f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model or "unknown",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def completion_to_sse(completion: dict[str, Any]) -> Iterator[str]:
    """Replay a cached completion as OpenAI-compatible SSE chunks."""
    message = (completion.get("choices") or [{}])[0].get("message") or {}
    content = message.get("content") or ""
    created = completion.get("created") or int(time.time())
    completion_id = completion.get("id") or f"chatcmpl-{created}"
    model = completion.get("model") or "unknown"
    finish_reason = (completion.get("choices") or [{}])[0].get("finish_reason") or "stop"

    words = content.split(" ") if content else [""]
    for index, word in enumerate(words):
        piece = word if index == len(words) - 1 else f"{word} "
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def is_successful_completion(response: dict[str, Any]) -> bool:
    choices = response.get("choices") or []
    if not choices:
        return False
    finish = choices[0].get("finish_reason")
    return finish in {"stop", "length"}
