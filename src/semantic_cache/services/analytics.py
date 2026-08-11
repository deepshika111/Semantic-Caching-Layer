from __future__ import annotations

from collections import Counter

from semantic_cache.models import CacheDecision


def analyze_thresholds(
    decisions: list[CacheDecision],
    candidates: list[float],
    request_type: str | None = None,
) -> dict:
    """Replay recorded similarities against candidate thresholds.

    Does not report a wrong-answer rate unless labelled feedback exists.
    """
    rows = [
        d
        for d in decisions
        if d.similarity is not None and (request_type is None or d.request_type == request_type)
    ]
    similarities = [d.similarity for d in rows if d.similarity is not None]
    labelled = [d for d in rows if d.feedback is not None]

    def _distribution(values: list[float]) -> dict[str, int]:
        buckets = {
            "0.00-0.70": 0,
            "0.70-0.85": 0,
            "0.85-0.90": 0,
            "0.90-0.92": 0,
            "0.92-0.95": 0,
            "0.95-0.97": 0,
            "0.97-0.99": 0,
            "0.99-1.00": 0,
        }
        for value in values:
            if value < 0.70:
                buckets["0.00-0.70"] += 1
            elif value < 0.85:
                buckets["0.70-0.85"] += 1
            elif value < 0.90:
                buckets["0.85-0.90"] += 1
            elif value < 0.92:
                buckets["0.90-0.92"] += 1
            elif value < 0.95:
                buckets["0.92-0.95"] += 1
            elif value < 0.97:
                buckets["0.95-0.97"] += 1
            elif value < 0.99:
                buckets["0.97-0.99"] += 1
            else:
                buckets["0.99-1.00"] += 1
        return buckets

    candidate_rows = []
    for threshold in candidates:
        would_hit = [d for d in rows if d.similarity is not None and d.similarity >= threshold]
        hit_feedback = [d.feedback for d in would_hit if d.feedback]
        candidate_rows.append(
            {
                "threshold": threshold,
                "candidate_matches": len(would_hit),
                "estimated_hit_rate": (len(would_hit) / len(rows)) if rows else 0.0,
                "labelled_correct": hit_feedback.count("correct"),
                "labelled_incorrect": hit_feedback.count("incorrect"),
            }
        )

    recommendation = None
    if len(labelled) >= 20:
        best = None
        for row in candidate_rows:
            incorrect = row["labelled_incorrect"]
            correct = row["labelled_correct"]
            total_labelled_hits = incorrect + correct
            if total_labelled_hits == 0:
                continue
            error_rate = incorrect / total_labelled_hits
            if error_rate > 0.05:
                continue
            score = (correct, -row["threshold"])
            if best is None or score > best[0]:
                best = (score, row["threshold"])
        if best:
            recommendation = {
                "threshold": best[1],
                "method": "offline_labelled_sweep",
                "note": "Requires labelled feedback; this is not online learning.",
            }

    return {
        "sample_size": len(rows),
        "labelled_sample_size": len(labelled),
        "similarity_distribution": _distribution(similarities),
        "candidates": candidate_rows,
        "recommended_threshold": recommendation,
    }


def summarize_near_misses(decisions: list[CacheDecision]) -> dict:
    near = [d for d in decisions if d.outcome == "near_miss"]
    similarities = [d.similarity for d in near if d.similarity is not None]
    types = Counter(d.request_type for d in near)
    return {
        "near_miss_count": len(near),
        "mean_similarity": (sum(similarities) / len(similarities)) if similarities else None,
        "min_similarity": min(similarities) if similarities else None,
        "max_similarity": max(similarities) if similarities else None,
        "request_types": dict(types),
        "note": "Lowering the threshold would turn these compatible near-misses into hits, increasing mismatch risk.",
    }
