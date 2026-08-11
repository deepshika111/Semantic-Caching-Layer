from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.requests_total = Counter(
            "semantic_cache_requests_total",
            "Total chat completion requests handled by the proxy",
            ["provider", "model", "request_type"],
            registry=self.registry,
        )
        self.hits_total = Counter(
            "semantic_cache_hits_total",
            "Cache hits",
            ["provider", "model", "request_type"],
            registry=self.registry,
        )
        self.misses_total = Counter(
            "semantic_cache_misses_total",
            "Cache misses that were forwarded to a provider",
            ["provider", "model", "request_type"],
            registry=self.registry,
        )
        self.near_misses_total = Counter(
            "semantic_cache_near_misses_total",
            "Compatible candidates below the hit threshold but within the near-miss band",
            ["provider", "model", "request_type"],
            registry=self.registry,
        )
        self.evictions_total = Counter(
            "semantic_cache_evictions_total",
            "Entries removed by capacity eviction or admin invalidation",
            ["reason"],
            registry=self.registry,
        )
        self.provider_requests_total = Counter(
            "semantic_cache_provider_requests_total",
            "Requests forwarded to an LLM provider",
            ["provider", "model"],
            registry=self.registry,
        )
        self.estimated_cost_saved_usd_total = Counter(
            "semantic_cache_estimated_cost_saved_usd_total",
            "Estimated USD saved by serving cache hits (not billed savings)",
            ["provider", "model"],
            registry=self.registry,
        )
        self.entries = Gauge(
            "semantic_cache_entries",
            "Approximate number of live cache entries",
            registry=self.registry,
        )
        self.similarity_score = Histogram(
            "semantic_cache_similarity_score",
            "Best-candidate cosine similarity",
            ["request_type"],
            buckets=(0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99, 1.0),
            registry=self.registry,
        )
        self.request_latency_seconds = Histogram(
            "semantic_cache_request_latency_seconds",
            "End-to-end proxy latency",
            ["outcome"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )
        self.provider_latency_seconds = Histogram(
            "semantic_cache_provider_latency_seconds",
            "Provider round-trip latency on cache misses",
            ["provider"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=self.registry,
        )
        self.hit_latency_seconds = Histogram(
            "semantic_cache_hit_latency_seconds",
            "Latency of cache-hit responses",
            buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
