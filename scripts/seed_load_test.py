#!/usr/bin/env python3
"""Send a small prompt set so the first Locust wave is not entirely cold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from load_tests.workload import (  # noqa: E402
    CLASSIFICATION_PROMPTS,
    EXACT_PROMPTS,
    PARAPHRASE_GROUPS,
    TEMPORAL_PROMPTS,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    prompts = list(EXACT_PROMPTS) + [group[0] for group in PARAPHRASE_GROUPS]
    prompts += TEMPORAL_PROMPTS + CLASSIFICATION_PROMPTS
    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        for prompt in prompts:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
            )
            response.raise_for_status()
            print(f"{response.headers.get('x-semantic-cache', '?'):4}  {prompt[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
