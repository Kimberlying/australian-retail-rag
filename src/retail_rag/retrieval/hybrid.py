from __future__ import annotations

from collections.abc import Sequence

from ..models import RetrievedChunk
from .base import rank
from .bm25 import BM25Retriever
from .dense import DenseRetriever


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, k: int = 60, weights: Sequence[float] | None = None
) -> dict[int, float]:
    """Fuse ranked lists of item indices: ``score(d) = sum_i w_i / (k + rank_i(d))``.

    RRF uses only ranks, so it combines retrievers whose raw scores live on
    incomparable scales (BM25 is unbounded, cosine is in [0, 1]) without any
    score normalisation or tuning. ``k=60`` is the value from Cormack et al. (2009).
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + position)
    return fused


class HybridRetriever:
    """BM25 (exact terms, numbers, codes) + dense (paraphrase) fused with RRF.

    Each candidate's ``relevance`` is its dense cosine similarity, so the
    evidence gate keeps a calibrated meaning even though ranking uses RRF.
    """

    name = "hybrid"

    def __init__(
        self,
        lexical: BM25Retriever,
        dense: DenseRetriever,
        *,
        rrf_k: int = 60,
        candidates: int = 20,
        weights: tuple[float, float] = (1.0, 1.0),
    ):
        if lexical.chunks != dense.chunks:
            raise ValueError("lexical and dense retrievers must index the same chunks")
        self.lexical = lexical
        self.dense = dense
        self.chunks = dense.chunks
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.weights = weights
        self.default_min_score = dense.default_min_score

    @staticmethod
    def _ranking(scores: Sequence[float], limit: int) -> list[int]:
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [index for index in order[:limit] if scores[index] > 0]

    def search(self, query: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        dense_scores = self.dense.score_all(query)
        fused = reciprocal_rank_fusion(
            [
                self._ranking(self.lexical.score_all(query), self.candidates),
                self._ranking(dense_scores, self.candidates),
            ],
            k=self.rrf_k,
            weights=self.weights,
        )
        scores = [fused.get(index, 0.0) for index in range(len(self.chunks))]
        return rank(self.chunks, scores, top_k=top_k, relevance=dense_scores)
