from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import Protocol, TypeVar

from ..models import DocumentChunk, RetrievedChunk
from .base import MetadataFilter, Retriever

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]], *, k: int = 60, weights: Sequence[float] | None = None
) -> dict[T, float]:
    """Fuse ranked lists of item ids: ``score(d) = sum_i w_i / (k + rank_i(d))``.

    RRF uses only ranks, so it combines retrievers whose raw scores live on
    incomparable scales (BM25 is unbounded, cosine is in [0, 1]) without any
    score normalisation or tuning. ``k=60`` is the value from Cormack et al. (2009).
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[T, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + position)
    return fused


class DenseSearch(Retriever, Protocol):
    """A dense retriever that can also score specific chunks (in memory or pgvector)."""

    def similarities(self, query: str, chunk_ids: Iterable[str]) -> dict[str, float]: ...


class HybridRetriever:
    """BM25 (exact terms, numbers, codes) + dense (paraphrase) fused with RRF.

    Each retriever contributes its top ``candidates``; fusion works on chunk ids,
    so the dense side can be an in-memory matrix or a pgvector HNSW index. Each
    result's ``relevance`` is its dense cosine similarity, so the evidence gate
    keeps a calibrated meaning even though ranking uses RRF.
    """

    name = "hybrid"

    def __init__(
        self,
        lexical: Retriever,
        dense: DenseSearch,
        *,
        rrf_k: int = 60,
        candidates: int = 20,
        weights: tuple[float, float] = (1.0, 1.0),
    ):
        if [chunk.chunk_id for chunk in lexical.chunks] != [c.chunk_id for c in dense.chunks]:
            raise ValueError("lexical and dense retrievers must index the same chunks")
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.weights = weights
        self.default_min_score = dense.default_min_score
        # Ties in fused score fall back to corpus order, so results are deterministic.
        self._position = {chunk.chunk_id: index for index, chunk in enumerate(lexical.chunks)}

    @property
    def chunks(self) -> list[DocumentChunk]:
        return self.lexical.chunks

    def search(
        self, query: str, *, top_k: int = 4, where: MetadataFilter | None = None
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        # Filters apply inside each retriever, so excluded chunks never take a candidate slot.
        lexical = self.lexical.search(query, top_k=self.candidates, where=where)
        dense = self.dense.search(query, top_k=self.candidates, where=where)
        fused = reciprocal_rank_fusion(
            [[item.chunk.chunk_id for item in lexical], [item.chunk.chunk_id for item in dense]],
            k=self.rrf_k,
            weights=self.weights,
        )
        best = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], self._position[chunk_id]))
        best = best[:top_k]

        chunk_by_id = {item.chunk.chunk_id: item.chunk for item in [*lexical, *dense]}
        relevance = {item.chunk.chunk_id: item.relevance for item in dense}
        missing = [chunk_id for chunk_id in best if relevance.get(chunk_id) is None]
        if missing:  # found by BM25 only: look up its dense similarity for the gate
            relevance.update(self.dense.similarities(query, missing))
        return [
            RetrievedChunk(
                chunk=chunk_by_id[chunk_id], score=fused[chunk_id], relevance=relevance[chunk_id]
            )
            for chunk_id in best
        ]
