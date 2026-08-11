from tests.helpers import make_settings

from semantic_cache.cache.policies import classify_request, evaluate_policy, threshold_for
from semantic_cache.models import ChatCompletionRequest, ChatMessage


def _req(content: str, **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role="user", content=content)],
        **kwargs,
    )


def test_classifies_temporal_factual_classification_creative() -> None:
    assert classify_request(_req("What is the weather today?")) == "temporal"
    assert classify_request(_req("What is Python?")) == "factual"
    assert classify_request(_req("Classify this review as positive or negative")) == "classification"
    assert classify_request(_req("Write a poem about the ocean")) == "creative"
    assert classify_request(_req("Hello there")) == "other"


def test_high_temperature_is_creative() -> None:
    assert classify_request(_req("Hello there", temperature=1.4)) == "creative"


def test_temporal_short_ttl() -> None:
    settings = make_settings()
    policy = evaluate_policy(_req("latest stock price for AAPL"), settings)
    assert policy.cacheable is True
    assert policy.ttl_seconds == settings.temporal_cache_ttl_seconds
    assert policy.request_type == "temporal"


def test_temporal_no_cache_mode() -> None:
    settings = make_settings(temporal_cache_mode="no_cache")
    policy = evaluate_policy(_req("current weather in Austin"), settings)
    assert policy.cacheable is False
    assert policy.reason == "temporal_no_cache"


def test_tools_and_n_are_not_cacheable() -> None:
    settings = make_settings()
    tools = evaluate_policy(_req("Hi", tools=[{"type": "function", "function": {"name": "x"}}]), settings)
    assert tools.cacheable is False
    multi = evaluate_policy(_req("Hi", n=3), settings)
    assert multi.cacheable is False


def test_threshold_selection() -> None:
    settings = make_settings()
    assert threshold_for("classification", settings) == 0.90
    assert threshold_for("factual", settings) == 0.95
    assert threshold_for("creative", settings) == 0.98
