from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

GeneratedBy = Literal["claude", "local", "local_fallback"]


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    retrievals: list[RetrievedChunk]
    generated_by: GeneratedBy = "local"
    refused: bool = False
    latency_ms: dict[str, float] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
