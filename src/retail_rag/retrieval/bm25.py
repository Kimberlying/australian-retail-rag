from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from ..models import DocumentChunk, RetrievedChunk
from .base import content_tokens, rank


class BM25Retriever:
    """Okapi BM25 over stopword-filtered tokens.

    BM25 adds term-frequency saturation (``k1``) and document-length
    normalisation (``b``) on top of IDF weighting. Its scores are unbounded, so
    it exposes no calibrated relevance and the evidence gate does not apply.
    """

    name = "bm25"
    default_min_score = 0.0

    def __init__(self, chunks: Iterable[DocumentChunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._term_freqs = [Counter(content_tokens(chunk.text)) for chunk in self.chunks]
        self._lengths = [sum(freqs.values()) for freqs in self._term_freqs]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for freqs in self._term_freqs:
            document_frequency.update(freqs.keys())
        total = len(self.chunks)
        # BM25+ style IDF that stays positive even for terms in most documents.
        self._idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score_all(self, query: str) -> list[float]:
        terms = [term for term in set(content_tokens(query)) if term in self._idf]
        scores = []
        for freqs, length in zip(self._term_freqs, self._lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / (self._avg_length or 1.0))
            score = 0.0
            for term in terms:
                tf = freqs.get(term, 0)
                if tf:
                    score += self._idf[term] * tf * (self.k1 + 1) / (tf + norm)
            scores.append(score)
        return scores

    def search(self, query: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        return rank(self.chunks, self.score_all(query), top_k=top_k)
