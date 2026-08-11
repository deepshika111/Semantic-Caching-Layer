from __future__ import annotations

import time

from fastapi.testclient import TestClient
from tests.helpers import make_settings

from semantic_cache.cache.key_builder import hash_system_prompt
from semantic_cache.main import create_app
from semantic_cache.models import ChatMessage
from semantic_cache.providers.base import ProviderError

ADMIN = {"X-Admin-Key": "test-admin-key"}


def _chat(client: TestClient, content: str, **extra):
    body = {
        "model": extra.pop("model", "gpt-4o-mini"),
        "messages": extra.pop("messages", [{"role": "user", "content": content}]),
        **extra,
    }
    response = client.post("/v1/chat/completions", json=body)
    return response, response.headers.get("x-semantic-cache")


def test_health_ready_metrics(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "semantic_cache_requests_total" in metrics.text


def test_exact_miss_then_semantic_hit(client: TestClient) -> None:
    first, status = _chat(client, "What is Python?")
    assert first.status_code == 200
    assert status == "MISS"
    assert "Mock answer" in first.json()["choices"][0]["message"]["content"]

    second, status = _chat(client, "Explain Python to me.")
    assert second.status_code == 200
    assert status == "HIT"
    assert (
        second.json()["choices"][0]["message"]["content"] == first.json()["choices"][0]["message"]["content"]
    )
    similarity = float(second.headers["x-semantic-cache-similarity"])
    assert similarity >= 0.95


def test_different_system_prompt_is_miss(client: TestClient) -> None:
    _chat(
        client,
        "What is Python?",
        messages=[
            {"role": "system", "content": "Answer as a professor."},
            {"role": "user", "content": "What is Python?"},
        ],
    )
    _second, status = _chat(
        client,
        "What is Python?",
        messages=[
            {"role": "system", "content": "Answer like a pirate."},
            {"role": "user", "content": "What is Python?"},
        ],
    )
    assert status == "MISS"


def test_incompatible_model_is_miss(client: TestClient) -> None:
    _chat(client, "What is Python?", model="gpt-4o-mini")
    _second, status = _chat(client, "What is Python?", model="gpt-4o")
    assert status == "MISS"


def test_different_temperature_is_miss(client: TestClient) -> None:
    _chat(client, "What is Python?", temperature=0.2)
    _second, status = _chat(client, "What is Python?", temperature=0.8)
    assert status == "MISS"


def test_expired_entry_is_miss(client: TestClient, app) -> None:
    first, status = _chat(client, "What is Python?")
    assert status == "MISS"
    repo = app.state.repository
    records = list(repo._records.values())
    assert records
    records[0].expires_at = time.time() - 1
    _second, status = _chat(client, "What is Python?")
    assert status == "MISS"


def test_failed_provider_response_is_not_cached(client: TestClient, app) -> None:
    class Boom:
        name = "mock"

        async def complete(self, request, api_key=None):
            raise ProviderError("upstream failed", status_code=502)

        async def stream(self, request, api_key=None):
            if False:
                yield ""
            raise ProviderError("upstream failed", status_code=502)

    app.state.proxy._router.register("mock", Boom())
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is Python?"}]},
    )
    assert response.status_code == 502
    assert len(app.state.repository._records) == 0


def test_admin_invalidation(client: TestClient) -> None:
    _chat(client, "What is Python?")
    assert client.get("/admin/cache/stats", headers=ADMIN).json()["entries"] == 1
    deleted = client.delete("/admin/cache/model/gpt-4o-mini", headers=ADMIN)
    assert deleted.json()["deleted"] == 1
    _second, status = _chat(client, "What is Python?")
    assert status == "MISS"


def test_admin_requires_key(client: TestClient) -> None:
    response = client.delete("/admin/cache/model/gpt-4o-mini")
    assert response.status_code == 401


def test_streaming_miss_then_hit(client: TestClient) -> None:
    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "stream": True,
            "messages": [{"role": "user", "content": "What is Python?"}],
        },
    )
    assert first.status_code == 200
    assert first.headers["x-semantic-cache"] == "MISS"
    assert "data: [DONE]" in first.text

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "stream": True,
            "messages": [{"role": "user", "content": "Explain Python to me."}],
        },
    )
    assert second.status_code == 200
    assert second.headers["x-semantic-cache"] == "HIT"
    assert "text/event-stream" in second.headers["content-type"]
    assert "data: [DONE]" in second.text


def test_cache_bypass_header() -> None:
    settings = make_settings()
    app = create_app(settings)
    with TestClient(app) as client:
        _chat(client, "What is Python?")
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is Python?"}]},
            headers={"X-Cache-Bypass": "true"},
        )
        assert response.headers["x-semantic-cache"] == "BYPASS"


def test_system_hash_invalidation(client: TestClient) -> None:
    messages = [
        ChatMessage(role="system", content="Be brief."),
        ChatMessage(role="user", content="What is Python?"),
    ]
    _chat(
        client,
        "What is Python?",
        messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is Python?"}],
    )
    digest = hash_system_prompt(messages)
    deleted = client.delete(f"/admin/cache/system/{digest}", headers=ADMIN)
    assert deleted.json()["deleted"] == 1
