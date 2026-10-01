from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

GeneratedBy = Literal["claude", "local", "local_fallback"]
Route = Literal["rag", "sql", "hybrid"]
RefusalReason = Literal["metadata_filter", "evidence_gate", "model"]


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
    route: Route = "rag"
    filters: dict[str, list[Any]] = field(default_factory=dict)
    """Metadata filters extracted from the question (company, fiscal_year)."""
    refusal_reason: RefusalReason | None = None
    sql_queries: list[str] = field(default_factory=list)
    """Statements the SQL agent executed, for transparency and debugging."""
