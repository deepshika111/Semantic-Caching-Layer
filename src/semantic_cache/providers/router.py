from __future__ import annotations

from semantic_cache.config import Settings
from semantic_cache.providers.anthropic_provider import AnthropicProvider
from semantic_cache.providers.base import LLMProvider, ProviderError
from semantic_cache.providers.mock_provider import MockProvider
from semantic_cache.providers.ollama_provider import OllamaProvider
from semantic_cache.providers.openai_provider import OpenAIProvider

OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-", "text-davinci")
ANTHROPIC_PREFIXES = ("claude-", "claude")
OLLAMA_PREFIXES = ("llama", "mistral", "qwen", "phi", "gemma", "ollama/")


class ProviderRouter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[str, LLMProvider] = {
            "openai": OpenAIProvider(settings),
            "anthropic": AnthropicProvider(settings),
            "ollama": OllamaProvider(settings),
            "mock": MockProvider(),
        }

    def resolve_name(self, model: str, explicit: str | None = None) -> str:
        if self._settings.provider_mode == "mock":
            return "mock"
        if explicit:
            return explicit.lower()
        lowered = model.lower()
        if lowered.startswith("openai/") or lowered.startswith(OPENAI_PREFIXES):
            return "openai"
        if lowered.startswith("anthropic/") or lowered.startswith(ANTHROPIC_PREFIXES):
            return "anthropic"
        if lowered.startswith(OLLAMA_PREFIXES):
            return "ollama"
        if self._settings.provider_mode in self._providers:
            return self._settings.provider_mode
        return self._settings.default_provider

    def register(self, name: str, provider: LLMProvider) -> None:
        self._providers[name] = provider

    def get(self, model: str, explicit: str | None = None) -> tuple[str, LLMProvider]:
        name = self.resolve_name(model, explicit)
        provider = self._providers.get(name)
        if provider is None:
            raise ProviderError(f"Unknown provider '{name}'", status_code=400)
        if name == "anthropic" and not self._settings.anthropic_api_key:
            # Adapter is present; it is only usable when configured.
            if explicit == "anthropic" or model.lower().startswith("claude"):
                raise ProviderError("Anthropic is not configured (set ANTHROPIC_API_KEY)", status_code=401)
        return name, provider
