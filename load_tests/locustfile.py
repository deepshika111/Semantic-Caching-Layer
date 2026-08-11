"""Locust workload for the semantic cache proxy.

Mix is configurable through environment variables. This is not a claim about
production traffic; it is a reproducible benchmark shape.

  EXACT_REPEAT      default 0.35
  PARAPHRASE        default 0.25
  UNIQUE            default 0.20
  TEMPORAL          default 0.10
  CLASSIFICATION    default 0.05
  CREATIVE          default 0.05

Set CACHE_BYPASS=true to send X-Cache-Bypass and measure the uncached baseline.
Set LOADTEST_REQUESTS=2000 to stop after N requests (used by run_benchmark.py).
"""

from __future__ import annotations

import os

from locust import HttpUser, between, events, task

from load_tests.workload import choose_prompt


class SemanticCacheUser(HttpUser):
    wait_time = between(0.0, 0.05)

    def on_start(self) -> None:
        self.model = os.getenv("LOADTEST_MODEL", "gpt-4o-mini")
        self.bypass = os.getenv("CACHE_BYPASS", "").lower() in {"1", "true", "yes"}

    @task
    def chat(self) -> None:
        prompt, kind = choose_prompt()
        headers = {"Content-Type": "application/json"}
        if self.bypass:
            headers["X-Cache-Bypass"] = "true"
        with self.client.post(
            "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            headers=headers,
            name=f"/v1/chat/completions [{kind}]",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}")


@events.request.add_listener
def _stop_at_target(environment, **_kwargs) -> None:
    target = int(os.getenv("LOADTEST_REQUESTS", "0") or 0)
    if target <= 0 or environment.runner is None:
        return
    if environment.runner.stats.total.num_requests >= target:
        environment.runner.quit()
