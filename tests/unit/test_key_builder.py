from semantic_cache.cache.key_builder import (
    build_fingerprint,
    build_semantic_input,
    hash_system_prompt,
    sha256_hex,
)
from semantic_cache.models import ChatCompletionRequest, ChatMessage


def _request(**overrides) -> ChatCompletionRequest:
    base = {
        "model": "gpt-4o-mini",
        "messages": [ChatMessage(role="user", content="What is Python?")],
    }
    base.update(overrides)
    return ChatCompletionRequest(**base)


def test_semantic_input_excludes_system_prompt() -> None:
    messages = [
        ChatMessage(role="system", content="You are a tutor."),
        ChatMessage(role="user", content="What is Python?"),
        ChatMessage(role="assistant", content="A language."),
        ChatMessage(role="user", content="Tell me more."),
    ]
    text = build_semantic_input(messages)
    assert "You are a tutor." not in text
    assert "user: What is Python?" in text
    assert "assistant: A language." in text
    assert "user: Tell me more." in text


def test_system_prompt_hash_normalizes_whitespace() -> None:
    left = [ChatMessage(role="system", content="Be   helpful\nplease")]
    right = [ChatMessage(role="system", content="Be helpful please")]
    assert hash_system_prompt(left) == hash_system_prompt(right)
    assert hash_system_prompt(left) == sha256_hex("Be helpful please")


def test_fingerprint_stable_for_equivalent_defaults() -> None:
    explicit = _request(temperature=1.0, top_p=1.0, n=1)
    implicit = _request()
    assert build_fingerprint(explicit, "openai") == build_fingerprint(implicit, "openai")


def test_fingerprint_changes_with_system_prompt() -> None:
    left = _request(messages=[ChatMessage(role="user", content="Hi")])
    right = _request(
        messages=[
            ChatMessage(role="system", content="Be terse."),
            ChatMessage(role="user", content="Hi"),
        ]
    )
    assert build_fingerprint(left, "openai") != build_fingerprint(right, "openai")


def test_fingerprint_changes_with_model_provider_and_temperature() -> None:
    base = _request()
    other_model = _request(model="gpt-4o")
    other_temp = _request(temperature=0.2)
    assert build_fingerprint(base, "openai") != build_fingerprint(other_model, "openai")
    assert build_fingerprint(base, "openai") != build_fingerprint(base, "mock")
    assert build_fingerprint(base, "openai") != build_fingerprint(other_temp, "openai")


def test_fingerprint_does_not_use_builtin_hash() -> None:
    value = build_fingerprint(_request(), "openai")
    assert len(value) == 64
    int(value, 16)
