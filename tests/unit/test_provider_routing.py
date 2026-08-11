from tests.helpers import make_settings

from semantic_cache.providers.router import ProviderRouter


def test_mock_mode_always_resolves_mock() -> None:
    router = ProviderRouter(make_settings(provider_mode="mock"))
    assert router.resolve_name("gpt-4o-mini") == "mock"
    assert router.resolve_name("claude-3-5-sonnet") == "mock"


def test_openai_and_anthropic_and_ollama_prefixes() -> None:
    router = ProviderRouter(make_settings(provider_mode="openai", anthropic_api_key="x"))
    assert router.resolve_name("gpt-4o-mini") == "openai"
    assert router.resolve_name("o3-mini") == "openai"
    assert router.resolve_name("claude-3-5-sonnet") == "anthropic"
    assert router.resolve_name("llama3") == "ollama"
    assert router.resolve_name("ollama/mistral") == "ollama"


def test_explicit_override_wins() -> None:
    router = ProviderRouter(make_settings(provider_mode="openai"))
    assert router.resolve_name("gpt-4o-mini", explicit="ollama") == "ollama"
