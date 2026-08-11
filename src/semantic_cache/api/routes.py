from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

from semantic_cache.embeddings.openai_embeddings import EmbeddingError
from semantic_cache.models import ChatCompletionRequest
from semantic_cache.providers.base import ProviderError
from semantic_cache.services.proxy_service import extract_bearer

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    repository = request.app.state.repository
    if not await repository.ping():
        raise HTTPException(status_code=503, detail="cache backend unavailable")
    return {"status": "ready", "backend": request.app.state.settings.cache_backend}


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    payload = request.app.state.metrics.render()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@router.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_cache_namespace: str = Header(default="default", alias="X-Cache-Namespace"),
    x_cache_bypass: str | None = Header(default=None, alias="X-Cache-Bypass"),
    x_llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
):
    proxy = request.app.state.proxy
    bypass = (x_cache_bypass or "").lower() in {"1", "true", "yes"}
    try:
        result = await proxy.handle(
            payload,
            tenant=x_cache_namespace or "default",
            bypass=bypass,
            provider_override=x_llm_provider,
            api_key=extract_bearer(authorization),
        )
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    if result.stream is not None:
        return StreamingResponse(
            result.stream,
            media_type="text/event-stream",
            headers=result.headers,
        )
    return JSONResponse(content=result.body, headers=result.headers)
