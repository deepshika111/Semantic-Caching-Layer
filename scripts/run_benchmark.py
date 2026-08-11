#!/usr/bin/env python3
"""Run comparable cache-enabled and cache-disabled load tests.

This script talks to an already-running proxy. Default mode is LOCAL/MOCK
benchmarking: it does not call paid providers unless the proxy is configured
to do so.

Results are written to load_tests/results/latest.json.

Never treat mock-provider savings as billed OpenAI savings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "load_tests" / "results"


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


def _parse_locust_stats(path: Path) -> dict:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    aggregated = next((row for row in rows if row.get("Name") == "Aggregated"), rows[-1] if rows else {})
    return aggregated


def _run_locust(base_url: str, requests: int, users: int, bypass: bool, label: str) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    prefix = RESULTS / label
    env = {
        **os.environ,
        "LOADTEST_REQUESTS": str(requests),
        "CACHE_BYPASS": "true" if bypass else "false",
    }
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(ROOT / "load_tests" / "locustfile.py"),
        "--headless",
        "-u",
        str(users),
        "-r",
        str(max(users, 1)),
        "--host",
        base_url,
        "--csv",
        str(prefix),
        "--only-summary",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"locust exited with {completed.returncode}")
    stats = _parse_locust_stats(Path(f"{prefix}_stats.csv"))
    return {"elapsed_seconds": elapsed, "locust": stats}


def _sample_proxy(base_url: str, requests: int, bypass: bool, model: str) -> dict:
    """Direct httpx mix used to capture cache headers Locust does not parse."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from load_tests.workload import choose_prompt

    from semantic_cache.monitoring.cost import estimate_cost_usd

    latencies: list[float] = []
    hit_latencies: list[float] = []
    miss_latencies: list[float] = []
    outcomes: dict[str, int] = {"HIT": 0, "MISS": 0, "SKIP": 0, "BYPASS": 0}
    saved = 0.0
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        for _ in range(requests):
            prompt, _kind = choose_prompt()
            headers = {}
            if bypass:
                headers["X-Cache-Bypass"] = "true"
            started = time.perf_counter()
            response = client.post(
                "/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                headers=headers,
            )
            elapsed = time.perf_counter() - started
            response.raise_for_status()
            status = response.headers.get("x-semantic-cache", "MISS")
            outcomes[status] = outcomes.get(status, 0) + 1
            latencies.append(elapsed)
            if status == "HIT":
                hit_latencies.append(elapsed)
                usage = response.json().get("usage") or {}
                saved += estimate_cost_usd(
                    model,
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                )
            else:
                miss_latencies.append(elapsed)

    total = sum(outcomes.values()) or 1
    hits = outcomes.get("HIT", 0)
    return {
        "total_requests": total,
        "hits": hits,
        "misses": outcomes.get("MISS", 0),
        "skips": outcomes.get("SKIP", 0),
        "bypasses": outcomes.get("BYPASS", 0),
        "hit_rate": hits / total,
        "miss_rate": outcomes.get("MISS", 0) / total,
        "p50_latency_seconds": _percentile(latencies, 50),
        "p95_latency_seconds": _percentile(latencies, 95),
        "p99_latency_seconds": _percentile(latencies, 99),
        "hit_latency_p50_seconds": _percentile(hit_latencies, 50),
        "miss_latency_p50_seconds": _percentile(miss_latencies, 50),
        "provider_calls_avoided": hits,
        "estimated_cost_saved_usd": saved,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--skip-locust", action="store_true")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=10.0) as client:
        health = client.get("/health")
        health.raise_for_status()

    print("Running cache-enabled sample...", flush=True)
    enabled = _sample_proxy(args.base_url, args.requests, bypass=False, model=args.model)
    print("Running cache-disabled baseline...", flush=True)
    disabled = _sample_proxy(args.base_url, args.requests, bypass=True, model=args.model)

    locust_enabled = None
    locust_disabled = None
    if not args.skip_locust:
        try:
            locust_enabled = _run_locust(
                args.base_url, args.requests, args.users, bypass=False, label="cache_on"
            )
            locust_disabled = _run_locust(
                args.base_url, args.requests, args.users, bypass=True, label="cache_off"
            )
        except FileNotFoundError:
            print("locust is not installed; skipped concurrent load. pip install '.[dev]'")
        except RuntimeError as exc:
            print(f"locust failed ({exc}); header-based sample is still valid.")

    payload = {
        "benchmark_kind": "LOCAL/MOCK unless the proxy was started with a real provider",
        "base_url": args.base_url,
        "requests_per_mode": args.requests,
        "cache_enabled": enabled,
        "cache_disabled": disabled,
        "locust_cache_enabled": locust_enabled,
        "locust_cache_disabled": locust_disabled,
        "comparison": {
            "hit_rate": enabled["hit_rate"],
            "provider_calls_avoided": enabled["provider_calls_avoided"],
            "p95_latency_ratio": (
                enabled["p95_latency_seconds"] / disabled["p95_latency_seconds"]
                if enabled["p95_latency_seconds"] and disabled["p95_latency_seconds"]
                else None
            ),
            "estimated_cost_saved_usd_from_proxy_counter": enabled["estimated_cost_saved_usd"],
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "latest.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["comparison"], indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
