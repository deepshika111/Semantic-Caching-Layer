import json

from semantic_cache.services.streaming import (
    assemble_openai_stream,
    completion_to_sse,
    is_successful_completion,
)


def test_assemble_requires_done_marker() -> None:
    chunks = [
        'data: {"id":"x","model":"gpt-4o-mini","choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
    ]
    assert assemble_openai_stream(chunks) is None


def test_assemble_complete_stream() -> None:
    chunks = [
        'data: {"id":"x","model":"gpt-4o-mini","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
        'data: {"id":"x","model":"gpt-4o-mini","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        "data: [DONE]\n\n",
    ]
    assembled = assemble_openai_stream(chunks)
    assert assembled is not None
    assert assembled["choices"][0]["message"]["content"] == "Hello"
    assert is_successful_completion(assembled)


def test_cached_completion_can_be_replayed_as_sse() -> None:
    completion = {
        "id": "x",
        "model": "gpt-4o-mini",
        "created": 1,
        "choices": [{"message": {"role": "assistant", "content": "Hi there"}, "finish_reason": "stop"}],
    }
    payload = "".join(completion_to_sse(completion))
    assert "data: [DONE]" in payload
    assert json.loads(payload.split("data: ")[1].split("\n")[0])["choices"][0]["delta"]["content"].startswith(
        "Hi"
    )
