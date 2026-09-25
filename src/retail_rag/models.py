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
    """Retriever-native ranking score (cosine, BM25, or RRF); only comparable within one run."""
    relevance: float | None = None
    """Calibrated query-chunk similarity in [0, 1], used for the evidence gate.

    ``None`` when the retriever has no calibrated signal (plain BM25), in which
    case the gate does not apply.
    """


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    retrievals: list[RetrievedChunk]
    generated_by: GeneratedBy = "local"
    refused: bool = False
    evidence_score: float | None = None
    """Best retrieval relevance before the evidence gate (None if uncalibrated)."""
    latency_ms: dict[str, float] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
