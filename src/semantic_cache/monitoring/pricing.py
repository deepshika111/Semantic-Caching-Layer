"""Estimated USD per 1 million tokens.

These figures are configuration, not invoices. Update this table when provider
pricing changes. Cache-hit savings are computed from stored token usage and
this table; they are estimates.
"""

from __future__ import annotations

# Last reviewed: 2026-08. Verify against the provider's current price list.
USD_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1-mini": {"input": 1.10, "output": 4.40},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}

DEFAULT_RATES = {"input": 0.50, "output": 1.50}


def rates_for_model(model: str) -> dict[str, float]:
    lowered = model.lower()
    if lowered in USD_PER_MILLION_TOKENS:
        return USD_PER_MILLION_TOKENS[lowered]
    for prefix, rates in USD_PER_MILLION_TOKENS.items():
        if lowered.startswith(prefix):
            return rates
    return DEFAULT_RATES
