from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_base_url: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    provider_mode: Literal["openai", "anthropic", "ollama", "mock"] = "mock"
    default_provider: str = "openai"

    redis_url: str = "redis://localhost:6379"
    cache_backend: Literal["redis", "memory"] = "redis"
    redis_index_name: str = "semantic_cache"
    redis_key_prefix: str = "scache"

    cache_enabled: bool = True
    semantic_cache_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    near_miss_delta: float = Field(default=0.05, ge=0.0, le=1.0)
    default_cache_ttl_seconds: int = 86400
    temporal_cache_ttl_seconds: int = 3600
    temporal_cache_mode: Literal["short_ttl", "no_cache"] = "short_ttl"
    creative_cache_enabled: bool = True
    creative_threshold: float = Field(default=0.98, ge=0.0, le=1.0)
    classification_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    factual_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    max_cache_entries: int = 100_000
    cache_search_top_k: int = 8

    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_timeout_seconds: float = 15.0

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    debug_cache_headers: bool = False
    prompt_retention_enabled: bool = False
    decision_log_max_entries: int = 10_000

    admin_api_key: str = ""

    provider_timeout_seconds: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()