from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[Any] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    Unknown fields are preserved and forwarded to the provider.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    n: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    seed: int | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class CacheRecord(BaseModel):
    cache_id: str
    tenant: str = "default"
    semantic_input: str = ""
    semantic_input_hash: str
    embedding: list[float]
    response: dict[str, Any]
    provider: str
    model: str
    fingerprint: str
    system_prompt_hash: str
    request_type: str
    tags: list[str] = Field(default_factory=list)
    created_at: float
    expires_at: float
    ttl_seconds: int
    hit_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


class SearchHit(BaseModel):
    record: CacheRecord
    similarity: float


class CacheDecision(BaseModel):
    decision_id: str
    timestamp: float
    tenant: str
    provider: str
    model: str
    request_type: str
    similarity: float | None = None
    threshold: float
    outcome: Literal["hit", "miss", "near_miss", "skip", "bypass"]
    cache_id: str | None = None
    fingerprint: str | None = None
    feedback: Literal["correct", "incorrect"] | None = None


class CachePolicyResult(BaseModel):
    cacheable: bool
    reason: str
    ttl_seconds: int
    threshold: float
    request_type: str
    tags: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    decision_id: str | None = None
    cache_id: str | None = None
    label: Literal["correct", "incorrect"]


class ThresholdAnalysisRequest(BaseModel):
    candidates: list[float] = Field(default_factory=lambda: [0.90, 0.92, 0.95, 0.97, 0.98])
    request_type: str | None = None
