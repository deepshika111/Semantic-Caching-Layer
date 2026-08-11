from __future__ import annotations

from semantic_cache.monitoring.pricing import rates_for_model


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = rates_for_model(model)
    input_cost = (prompt_tokens / 1_000_000) * rates["input"]
    output_cost = (completion_tokens / 1_000_000) * rates["output"]
    return input_cost + output_cost
