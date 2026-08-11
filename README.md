# Semantic Caching Layer for LLM APIs

A caching proxy for LLM APIs that matches on meaning, not exact text. Point an OpenAI-compatible client at it   paraphrased prompts can hit the cache, exact ones definitely will, and everything else gets forwarded to the real provider.

In a 2,000-request mock benchmark it avoided 79.1% of provider calls (hit P50 2.8ms vs. 13.6ms). P95 got worse in that same run   more on why below, it's not a bug.

## Problem

LLM calls cost money per token and add real latency. A lot of traffic is the same few questions asked in different words. Exact-match caches miss all of that, so every rewording goes upstream even when the answer's already sitting there.

## How it works

1. Classify the request   cacheable? which threshold and TTL?
2. Embed the actual conversational content, not the raw request JSON.
3. Search Redis for similar vectors that also match a compatibility fingerprint.
4. If cosine similarity clears the threshold, serve the cached response.
5. Otherwise, call the provider, stream to the client as it comes in, and cache it only if it finishes cleanly.

Default threshold is 0.95   conservative on purpose. See **Real-provider verification** for the data behind that number.

## Architecture

```
flowchart TD
    app[Application] --> proxy[FastAPI proxy]
    proxy --> policy[Cache policy]
    policy -->|not cacheable| provider[Provider]
    policy --> embed[Embedding service]
    embed --> redis[Redis vector search]
    redis -->|similarity >= threshold + fingerprint match| hit[Cached response]
    redis -->|miss / near-miss| provider
    provider --> client[Client]
    provider -->|complete success| redis
    hit --> client
```

## Cache safety

Similar wording isn't enough. A cached answer only gets served when the provider, model, system prompt hash, and generation parameters (temperature, top_p, max tokens, seed, penalties, etc.) all match. Otherwise a cheaper or differently-instructed model could quietly answer something it wasn't asked. `X-Cache-Namespace` is a tag for future tenant isolation   not a security boundary today.

## Streaming

On a miss, chunks stream to the client as they arrive while a separate buffer reconstructs the full response. Nothing gets cached until the stream finishes cleanly with a real `finish_reason`   dropped connections, provider errors, and cancelled streams all skip the cache write.

On a hit, the stored response replays as SSE so a streaming client doesn't get a plain JSON blob back.

## Cache policies

| Type | Threshold | TTL |
|---|---|---|
| classification | 0.90 | 24h |
| factual | 0.95 | 24h |
| other | 0.95 | 24h |
| creative | 0.98 | 24h or disabled |
| temporal | 0.95 | 1h or no-cache |

Starting points, not tuned optima. Temporal language ("today," "latest," "weather") gets a short TTL since the same question can have a different right answer an hour later. Creative prompts get a stricter threshold because "similar" is weaker when the whole point is generating something new.

Admin invalidation (API-key protected): `DELETE /admin/cache/model/{model}`, `/system/{hash}`, `/tag/{tag}`.

## Real-provider verification

The mock benchmark below tests the proxy's own overhead, not real embedding behavior. So I ran a small check against live infrastructure   Docker Compose, Redis Stack, real `text-embedding-3-small` and `gpt-4o-mini` via OpenRouter.

| Prompt pair | Similarity | At 0.95 | At 0.70 |
|---|---|---|---|
| Exact repeat | 0.999999 | HIT | HIT |
| "What is Python?" vs. "Explain Python to me." | 0.741073 | MISS | HIT |
| "What is Python?" vs. "Can you tell me about the Python programming language?" | 0.759145 | MISS | HIT |

Real paraphrases scored 0.74–0.76, not 0.95+. At the default threshold, only near-exact wording actually hits. Small sample, but it's real data   enough to say 0.95 is tuned for duplicates, not paraphrases.

Two bugs turned up doing this, neither caught by the test suite since both only show up against a real endpoint:

1. The OpenAI embedding and chat clients didn't pass `base_url`, so anything other than `api.openai.com` got silently ignored   it failed as an auth error, not a routing one.
2. `docker-compose.yml` listed `OPENAI_API_KEY` in the container's `environment:` block but not `OPENAI_BASE_URL`. Compose doesn't forward `.env` wholesale   only what's explicitly named there reaches the container.

## Monitoring

`GET /metrics` exposes hit/miss/near-miss counters, eviction and provider-call counts, estimated cost saved, and latency histograms for request/provider/hit paths. Labels are limited to provider, model, and request type   no prompt text or IDs, to keep cardinality sane.

Grafana comes up provisioned automatically at [localhost:3000](http://localhost:3000) (admin/admin) after `docker compose up`.

There's also a threshold-analysis endpoint and a near-miss summary for looking at similarity distributions   neither reports a wrong-answer rate unless someone's actually labelled results via the feedback endpoint.

## Benchmarks

From `python scripts/run_benchmark.py --requests 2000 --skip-locust`. This run was **local/mock**: `PROVIDER_MODE=mock`, in-memory backend, hashed embeddings, a ~10ms mock provider. Not real quality, not real cost.

Mix: ~35% exact repeats, 25% paraphrases, 20% unique, 10% temporal, 5% classification, 5% creative.

| Metric | Disabled | Enabled | Diff |
|---|---|---|---|
| Hit rate | 0% | 79.1% | +79.1pp |
| Provider calls avoided |   | 1,582 / 2,000 |   |
| P50 | 13.6ms | 3.4ms | −75% |
| P95 | 15.5ms | 46.5ms | +200% |
| Hit P50 |   | 2.8ms |   |
| Est. cost saved | $0 | $0.0079 |   |

P95 got worse because a miss still has to embed and search on top of the mock provider's ~10ms, while the disabled baseline just waits that flat 10ms. Against a real LLM (200ms–2s), the 2.8ms hit path would dominate instead of losing. Redis HNSW is the real production search path   the in-memory backend here is just for this benchmark and for tests.

A real-provider version of this same benchmark needs a funded API key and hasn't been run at 2,000-request scale yet.

## Cost analysis

A replaceable pricing table (`monitoring/pricing.py`) gets multiplied against cached token usage on each hit. Estimates, not invoices   they drift as vendors reprice, and mock-provider runs obviously aren't real billing.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Without Redis Stack:
```bash
CACHE_BACKEND=memory PROVIDER_MODE=mock make run
```

With Redis Stack on localhost:
```bash
CACHE_BACKEND=redis REDIS_URL=redis://localhost:6379 PROVIDER_MODE=mock make run
```

## Docker

```bash
docker compose up --build
```

Brings up the API (`:8000`), Redis Stack (`:6379`), Prometheus (`:9090`), Grafana (`:3000`). Keys come from the environment, never hardcoded. One gotcha: every variable the container needs has to be explicitly listed in the `api` service's `environment:` block in `docker-compose.yml`   dropping a value into `.env` alone won't reach the container (see the bug above).

Optional Ollama: `docker compose --profile ollama up --build`

Health checks: `GET /health` (process alive), `GET /ready` (cache backend actually responds).

## API usage

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"What is Python?"}]}'
```

`X-Semantic-Cache` header comes back `HIT`, `MISS`, `SKIP`, or `BYPASS`. Set `DEBUG_CACHE_HEADERS=true` for the similarity score too.

Works with the OpenAI Python client by just swapping `base_url`. Covers chat completions only   no images, assistants, or files endpoints, and no claim of full API compatibility.

## Testing

```bash
make test      # or: PYTHONPATH=src pytest -q
make lint
```

Unit tests cover fingerprinting, hashing, TTL/classification, thresholds, cost math, and invalidation. Integration tests run FastAPI end-to-end against the mock provider   semantic hit, system-prompt miss, model miss, param miss, expiry, failed-provider non-caching, streaming.

## Load testing

```bash
CACHE_BACKEND=memory PROVIDER_MODE=mock make run
python scripts/run_benchmark.py --requests 2000
# or, for concurrency:
locust -f load_tests/locustfile.py --headless -u 20 -r 20 --host http://127.0.0.1:8000
```

Traffic mix is configurable in the Locust file. Mock numbers shouldn't get presented as real-provider results.

## Design decisions

**Semantic vs. exact caching**   exact-match misses most of the real opportunity, since actual traffic rarely repeats verbatim.

**Threshold**   0.95 by default, biased toward missing a valid match over serving a wrong one. Checked against real embeddings (see above): 0.95 turns out to mean near-duplicate, not paraphrase. That's a deliberate tradeoff made with data, not a guess.

**Fingerprinting**   embeddings capture what was said; the fingerprint captures whether the answer's still valid. SHA-256 over canonical JSON, never `hash()`.

**Streaming**   nobody should wait for a full generation just because it might get cached afterward.

**Redis Stack**   HNSW cosine search and tag filtering in one process, via redis-py's async RediSearch client, so FastAPI never blocks on it.

**Provider isolation**   adapters normalize OpenAI/Anthropic/Ollama into one shape; cache keys include provider + model so answers can't leak across backends. An OpenAI-compatible endpoint like OpenRouter is just a config change, confirmed by actually routing real traffic through it.

**Mock mode**   hashed embeddings and a deterministic provider make CI and the 2k-request benchmark free. They test the proxy, not embedding quality   that's what the real-provider check is for.

## Tradeoffs

Higher threshold = safer, lower hit rate, more provider spend. Lower threshold = more hits, more risk a "close" prompt wasn't really asking the same thing. Short TTLs on temporal prompts trade hit rate for freshness. Caching creative output means replaying a past invention   sometimes wanted, sometimes not; default is "cache it, but only above 0.98." Storing raw prompts helps debugging and hurts privacy   `PROMPT_RETENTION_ENABLED=false` keeps a hash instead.

## Limitations

- Mock embeddings are bag-of-tokens, not real `text-embedding-3-small`. Three prompt pairs checked against the real thing   not a systematic evaluation.
- Anthropic and Ollama adapters exist but aren't as tested as the OpenAI path.
- No tenant isolation, no encryption at rest, no per-user ACL.
- Redis TTL expiry isn't fully reflected in the eviction metric.
- Chat completions only   tool calls are refused, not cached.
- Threshold "recommendation" is an offline sweep, not online learning.
- Request-type classification is heuristic and can be inconsistent   "what is X" and "tell me about X" classified differently during testing.

## Future improvements

Stronger semantic-equivalence checking backed by a real, larger paraphrase set. Distributed cache coordination. Feedback-driven threshold tuning that's actually used, not just offline. Cache warming. Real tenant isolation and encryption. Token-level streaming replay.

## Security / privacy

Keys and admin tokens never get logged. Admin routes need `X-Admin-Key`, checked with constant-time comparison. Prompts aren't stored unless `PROMPT_RETENTION_ENABLED=true`. Shared namespaces can leak answers across users   not safe for multi-tenant production without more isolation than exists today.

## Configuration

Key variables (full list in `.env.example`):

| Variable | Default | Role |
|---|---|---|
| `PROVIDER_MODE` | `mock` | openai / anthropic / ollama / mock |
| `OPENAI_API_KEY` | empty | embeddings + OpenAI (or compatible endpoints) |
| `OPENAI_BASE_URL` | empty | override for compatible endpoints   must also be listed in `docker-compose.yml` |
| `REDIS_URL` | `redis://localhost:6379` | |
| `CACHE_BACKEND` | `redis` | redis or memory |
| `SEMANTIC_CACHE_THRESHOLD` | `0.95` | see Real-provider verification |
| `DEFAULT_CACHE_TTL_SECONDS` | `86400` | |
| `ADMIN_API_KEY` | required for admin routes | |
| `PROMPT_RETENTION_ENABLED` | `false` | |
| `DEBUG_CACHE_HEADERS` | `false` | emits similarity score |