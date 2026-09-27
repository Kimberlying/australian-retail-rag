"""Second-stage reranking with a cross-encoder.

First-stage retrievers (BM25, bi-encoder embeddings) score the query and each
chunk independently, which is what makes them fast enough to scan a corpus. A
cross-encoder reads the query and one chunk *together*, so it can model their
interaction ("someone stealing" vs "a suspected shoplifter"), but it costs one
model call per pair. The standard compromise is retrieve-then-rerank: take the
top ``candidates`` from a cheap retriever and let the cross-encoder reorder them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from ..models import DocumentChunk, RetrievedChunk
from .base import MetadataFilter, Retriever

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return one relevance logit per text (higher is more relevant)."""
        ...


class FastEmbedReranker:
    """Local ONNX cross-encoder via fastembed (default: ms-marco-MiniLM-L-6-v2, ~80 MB)."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        *,
        model_path: Path | None = None,
        cache_dir: Path | None = None,
    ):
        try:
            from fastembed.rerank.cross_encoder import (  # noqa: PLC0415 - optional dependency
                TextCrossEncoder,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "Reranking needs the 'embeddings' extra: uv sync --extra embeddings"
            ) from exc

        self.model_name = model_name
        self._model = TextCrossEncoder(
            model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
            specific_model_path=str(model_path) if model_path else None,
        )

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(value) for value in self._model.rerank(query, list(texts))]


class RerankingRetriever:
    """Wrap any retriever: fetch ``candidates`` chunks, reorder them with a cross-encoder.

    Ranking uses the cross-encoder score, but ``relevance`` (the evidence-gate
    signal) is passed through from the first stage. Cross-encoder logits are
    unbounded and uncalibrated, and swapping the reranker must not silently move
    the refusal threshold.
    """

    def __init__(self, base: Retriever, reranker: Reranker, *, candidates: int = 20):
        if candidates < 1:
            raise ValueError("candidates must be positive")
        self.base = base
        self.reranker = reranker
        self.candidates = candidates
        self.name = f"{base.name}+rerank"
        self.default_min_score = base.default_min_score

    @property
    def chunks(self) -> list[DocumentChunk]:
        return self.base.chunks

    def search(
        self, query: str, *, top_k: int = 4, where: MetadataFilter | None = None
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        pool = self.base.search(query, top_k=max(top_k, self.candidates), where=where)
        if len(pool) <= 1:
            return pool[:top_k]
        scores = self.reranker.score(query, [item.chunk.text for item in pool])
        # sorted() is stable, so ties keep the first-stage order.
        order = sorted(range(len(pool)), key=lambda index: scores[index], reverse=True)
        return [
            RetrievedChunk(
                chunk=pool[index].chunk, score=scores[index], relevance=pool[index].relevance
            )
            for index in order[:top_k]
        ]
