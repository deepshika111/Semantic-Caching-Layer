from tests.helpers import make_settings

from semantic_cache.cache.semantic_cache import SemanticCache
from semantic_cache.models import CacheDecision
from semantic_cache.monitoring.cost import estimate_cost_usd
from semantic_cache.services.analytics import analyze_thresholds, summarize_near_misses


def test_cost_uses_pricing_table() -> None:
    cost = estimate_cost_usd("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert abs(cost - (0.15 + 0.60)) < 1e-9


def test_unknown_model_uses_default_rates() -> None:
    cost = estimate_cost_usd("mystery-model", 1_000_000, 0)
    assert cost == 0.50


def test_near_miss_classification() -> None:
    cache = SemanticCache(
        repository=None, settings=make_settings(semantic_cache_threshold=0.95, near_miss_delta=0.05)
    )  # type: ignore[arg-type]
    assert cache.classify_outcome(0.97, 0.95) == "hit"
    assert cache.classify_outcome(0.93, 0.95) == "near_miss"
    assert cache.classify_outcome(0.80, 0.95) == "miss"
    assert cache.classify_outcome(None, 0.95) == "miss"


def test_threshold_analysis_reports_hit_rate_not_wrong_answers() -> None:
    decisions = [
        CacheDecision(
            decision_id=str(i),
            timestamp=0,
            tenant="default",
            provider="mock",
            model="gpt-4o-mini",
            request_type="factual",
            similarity=score,
            threshold=0.95,
            outcome="miss",
        )
        for i, score in enumerate([0.91, 0.94, 0.96, 0.99])
    ]
    result = analyze_thresholds(decisions, [0.90, 0.95, 0.98])
    assert result["sample_size"] == 4
    by_threshold = {row["threshold"]: row for row in result["candidates"]}
    assert by_threshold[0.90]["candidate_matches"] == 4
    assert by_threshold[0.95]["candidate_matches"] == 2
    assert "wrong-answer" not in str(result).lower()
    assert result["recommended_threshold"] is None


def test_near_miss_summary() -> None:
    decisions = [
        CacheDecision(
            decision_id="1",
            timestamp=0,
            tenant="default",
            provider="mock",
            model="gpt-4o-mini",
            request_type="factual",
            similarity=0.93,
            threshold=0.95,
            outcome="near_miss",
        )
    ]
    summary = summarize_near_misses(decisions)
    assert summary["near_miss_count"] == 1
    assert summary["request_types"]["factual"] == 1
