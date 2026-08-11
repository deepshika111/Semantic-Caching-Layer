from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from semantic_cache.api.auth import require_admin
from semantic_cache.models import FeedbackRequest, ThresholdAnalysisRequest
from semantic_cache.services.analytics import analyze_thresholds, summarize_near_misses

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


@router.delete("/cache/model/{model}")
async def invalidate_model(model: str, request: Request) -> dict[str, int | str]:
    deleted = await request.app.state.cache.invalidate_model(model)
    request.app.state.metrics.evictions_total.labels(reason="admin_model").inc(deleted)
    return {"deleted": deleted, "model": model}


@router.delete("/cache/system/{system_hash}")
async def invalidate_system(system_hash: str, request: Request) -> dict[str, int | str]:
    deleted = await request.app.state.cache.invalidate_system(system_hash)
    request.app.state.metrics.evictions_total.labels(reason="admin_system").inc(deleted)
    return {"deleted": deleted, "system_prompt_hash": system_hash}


@router.delete("/cache/tag/{tag}")
async def invalidate_tag(tag: str, request: Request) -> dict[str, int | str]:
    deleted = await request.app.state.cache.invalidate_tag(tag)
    request.app.state.metrics.evictions_total.labels(reason="admin_tag").inc(deleted)
    return {"deleted": deleted, "tag": tag}


@router.get("/cache/stats")
async def cache_stats(request: Request) -> dict:
    count = await request.app.state.cache.entry_count()
    request.app.state.metrics.entries.set(count)
    return {"entries": count}


@router.post("/cache/feedback")
async def submit_feedback(payload: FeedbackRequest, request: Request) -> dict[str, int]:
    updated = await request.app.state.cache.set_feedback(
        payload.label,
        decision_id=payload.decision_id,
        cache_id=payload.cache_id,
    )
    return {"updated": updated}


@router.post("/analytics/thresholds")
async def threshold_analysis(payload: ThresholdAnalysisRequest, request: Request) -> dict:
    decisions = await request.app.state.cache.recent_decisions(limit=10_000)
    return analyze_thresholds(decisions, payload.candidates, payload.request_type)


@router.get("/analytics/near-misses")
async def near_miss_summary(request: Request) -> dict:
    decisions = await request.app.state.cache.recent_decisions(limit=10_000)
    return summarize_near_misses(decisions)
