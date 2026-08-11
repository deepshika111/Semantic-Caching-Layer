from semantic_cache.config import Settings


def make_settings(**overrides) -> Settings:
    values = {
        "cache_backend": "memory",
        "provider_mode": "mock",
        "admin_api_key": "test-admin-key",
        "cache_enabled": True,
        "prompt_retention_enabled": True,
        "debug_cache_headers": True,
        "embedding_dimensions": 256,
        "semantic_cache_threshold": 0.95,
        "near_miss_delta": 0.05,
        "default_cache_ttl_seconds": 86400,
        "temporal_cache_ttl_seconds": 3600,
        "creative_cache_enabled": True,
        "openai_api_key": "",
        "anthropic_api_key": "",
    }
    values.update(overrides)
    return Settings(**values)
