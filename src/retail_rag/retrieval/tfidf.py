from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from ..models import DocumentChunk, RetrievedChunk
from .base import load_chunk_file, rank, save_chunks, tokenize


class TfidfRetriever:
    """Sparse TF-IDF cosine retriever: the dependency-free baseline.

    Kept deliberately unchanged (no stopwords, no stemming) so it remains a
    stable reference point in the evaluation reports.
    """

    name = "tfidf"
    default_min_score = 0.05

    def __init__(self, chunks: Iterable[DocumentChunk]):
        self.chunks = list(chunks)
        self._document_frequency: Counter[str] = Counter()
        self._vectors: list[dict[str, float]] = []
        self._build()

    def _build(self) -> None:
        for chunk in self.chunks:
            terms = set(tokenize(chunk.text))
            self._document_frequency.update(terms)

        total = max(len(self.chunks), 1)
        idf = {
            term: math.log((1 + total) / (1 + frequency)) + 1
            for term, frequency in self._document_frequency.items()
        }
        self._vectors = []
        for chunk in self.chunks:
            counts = Counter(tokenize(chunk.text))
            total_terms = max(sum(counts.values()), 1)
            vector = {term: (count / total_terms) * idf[term] for term, count in counts.items()}
            self._vectors.append(vector)

    def _query_vector(self, query: str) -> dict[str, float]:
        counts = Counter(tokenize(query))
        total_terms = max(sum(counts.values()), 1)
        total = max(len(self.chunks), 1)
        vector: dict[str, float] = {}
        for term, count in counts.items():
            if term not in self._document_frequency:
                continue
            idf = math.log((1 + total) / (1 + self._document_frequency[term])) + 1
            vector[term] = (count / total_terms) * idf
        return vector

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(value * right.get(term, 0.0) for term, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def score_all(self, query: str) -> list[float]:
        query_vector = self._query_vector(query)
        return [self._cosine(query_vector, vector) for vector in self._vectors]

    def search(self, query: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        scores = self.score_all(query)
        return rank(self.chunks, scores, top_k=top_k, relevance=scores)

    def save(self, path: Path) -> None:
        save_chunks(self.chunks, path)

    @classmethod
    def load(cls, path: Path) -> TfidfRetriever:
        return cls(load_chunk_file(path))
