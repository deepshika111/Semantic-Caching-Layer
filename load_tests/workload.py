"""Shared prompt mix for Locust and the header-based benchmark script."""

from __future__ import annotations

import os
import random

EXACT_PROMPTS = [
    "What is Python?",
    "Explain HTTP status code 404.",
    "What does REST stand for?",
    "Define a binary search tree.",
    "How does a hash table work?",
    "What is the CAP theorem?",
    "Explain TCP versus UDP.",
    "What is a primary key?",
]

PARAPHRASE_GROUPS = [
    ["What is Python?", "Explain Python to me.", "Define Python.", "Tell me about Python."],
    ["What is a mutex?", "Explain mutexes.", "Define a mutex.", "Tell me about mutexes."],
    [
        "How does garbage collection work?",
        "Explain garbage collection.",
        "What is garbage collection?",
    ],
    ["What is Kubernetes?", "Explain Kubernetes to me.", "Define Kubernetes."],
    ["What is a deadlock?", "Explain deadlocks.", "Tell me about deadlocks."],
]

TEMPORAL_PROMPTS = [
    "What is the latest Python release today?",
    "Current weather in Austin",
    "Give me recent news about Redis",
    "What is the stock price right now?",
]

CLASSIFICATION_PROMPTS = [
    "Classify this review as positive or negative: the API was slow but accurate.",
    "Is this spam? yes or no: Congratulations you won a prize.",
    "Label this text as factual or opinion: Redis is the best database.",
]

CREATIVE_PROMPTS = [
    "Write a poem about cache invalidation.",
    "Write a short story about a load balancer.",
    "Imagine a world where every API call is free.",
]

UNIQUE_STEMS = [
    "Explain the difference between {a} and {b}.",
    "What is {a} used for in distributed systems?",
    "Give a concise overview of {a}.",
]

UNIQUE_TERMS = [
    "Raft",
    "Paxos",
    "consistent hashing",
    "bloom filters",
    "LSM trees",
    "write-ahead logs",
    "vector clocks",
    "CRDTs",
    "HNSW",
    "inverted indexes",
]


def mix_weights() -> dict[str, float]:
    keys = ["exact", "paraphrase", "unique", "temporal", "classification", "creative"]
    raw = {
        "exact": float(os.getenv("EXACT_REPEAT", "0.35")),
        "paraphrase": float(os.getenv("PARAPHRASE", "0.25")),
        "unique": float(os.getenv("UNIQUE", "0.20")),
        "temporal": float(os.getenv("TEMPORAL", "0.10")),
        "classification": float(os.getenv("CLASSIFICATION", "0.05")),
        "creative": float(os.getenv("CREATIVE", "0.05")),
    }
    total = sum(raw.values()) or 1.0
    return {key: raw[key] / total for key in keys}


def choose_prompt() -> tuple[str, str]:
    mix = mix_weights()
    kind = random.choices(list(mix), weights=list(mix.values()), k=1)[0]
    if kind == "exact":
        return random.choice(EXACT_PROMPTS), kind
    if kind == "paraphrase":
        return random.choice(random.choice(PARAPHRASE_GROUPS)), kind
    if kind == "temporal":
        return random.choice(TEMPORAL_PROMPTS), kind
    if kind == "classification":
        return random.choice(CLASSIFICATION_PROMPTS), kind
    if kind == "creative":
        return random.choice(CREATIVE_PROMPTS), kind
    left, right = random.sample(UNIQUE_TERMS, 2)
    stem = random.choice(UNIQUE_STEMS)
    return stem.format(a=left, b=right) + f" [{random.randint(1, 10_000_000)}]", kind
