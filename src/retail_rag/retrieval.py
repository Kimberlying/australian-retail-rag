from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .models import DocumentChunk, RetrievedChunk


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:['-][A-Za-z0-9_]+)*|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class TfidfRetriever:
    """A small persistent sparse-vector retriever for the first project version."""

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
            vector = {
                term: (count / total_terms) * idf[term]
                for term, count in counts.items()
            }
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

    def search(self, query: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        query_vector = self._query_vector(query)
        scored = [
            RetrievedChunk(chunk=chunk, score=self._cosine(query_vector, vector))
            for chunk, vector in zip(self.chunks, self._vectors)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return [item for item in scored[:top_k] if item.score > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in self.chunks
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TfidfRetriever":
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk(**item) for item in payload]
        return cls(chunks)
