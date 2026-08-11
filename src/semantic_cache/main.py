from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from semantic_cache.api.admin import router as admin_router
from semantic_cache.api.routes import router
from semantic_cache.cache.repository import build_repository
from semantic_cache.cache.semantic_cache import SemanticCache
from semantic_cache.config import Settings, get_settings
from semantic_cache.embeddings.mock_embeddings import MockEmbeddingService
from semantic_cache.embeddings.openai_embeddings import OpenAIEmbeddingService
from semantic_cache.monitoring.metrics import Metrics
from semantic_cache.providers.router import ProviderRouter
from semantic_cache.services.proxy_service import ProxyService


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def build_embeddings(settings: Settings):
    if settings.provider_mode == "mock" or not settings.openai_api_key:
        return MockEmbeddingService(dimensions=settings.embedding_dimensions)
    return OpenAIEmbeddingService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    _configure_logging(settings.log_level)
    repository = await build_repository(settings)
    embeddings = build_embeddings(settings)
    metrics = Metrics()
    cache = SemanticCache(repository, settings)
    router_ = ProviderRouter(settings)
    proxy = ProxyService(settings, cache, embeddings, router_, metrics)
    app.state.repository = repository
    app.state.embeddings = embeddings
    app.state.metrics = metrics
    app.state.cache = cache
    app.state.proxy = proxy
    try:
        yield
    finally:
        close = getattr(embeddings, "close", None)
        if close is not None:
            await close()
        await repository.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    app = FastAPI(
        title="Semantic LLM Cache",
        version="0.1.0",
        description="Provider-aware semantic caching proxy for OpenAI-compatible chat completions.",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.include_router(router)
    app.include_router(admin_router)
    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "semantic_cache.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
