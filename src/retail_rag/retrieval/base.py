"""Shared retriever contract, tokenisation, and chunk persistence."""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from ..models import DocumentChunk, RetrievedChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:['-][A-Za-z0-9_]+)*|[\u4e00-\u9fff]")

# Small English stopword list for BM25. Without it, a query such as "What is the
# capital of France?" matches every chunk through "what/is/the/of".
STOPWORDS = frozenset(
    """a about above after again all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few
    for from further had has have having he her here hers him his how i if in into is
    it its itself just me more most my no nor not now of off on once only or other our
    ours out over own same she should so some such than that the their theirs them then
    there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours""".split()  # noqa: SIM905
)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


MetadataFilter = Mapping[str, Collection[Any]]
"""``{key: allowed values}``. A chunk passes when, for every key, its metadata has
that key with one of the allowed values (list-valued metadata: any of them).
A chunk *without* the key is excluded, unless ``None`` is among the allowed
values: ``{"fiscal_year": [2024, None]}`` keeps undated documents (policies)
while dropping documents about other years."""


def matches_filter(chunk: DocumentChunk, where: MetadataFilter | None) -> bool:
    if not where:
        return True
    for key, allowed in where.items():
        if key not in chunk.metadata:
            if None in allowed:
                continue
            return False
        value = chunk.metadata[key]
        values = value if isinstance(value, list) else [value]
        if not any(item in allowed for item in values):
            return False
    return True


def apply_filter(
    chunks: Sequence[DocumentChunk], scores: Sequence[float], where: MetadataFilter | None
) -> list[float]:
    """Zero the score of every chunk that fails ``where`` so ``rank`` drops it."""
    if not where:
        return list(scores)
    return [
        score if matches_filter(chunk, where) else 0.0
        for chunk, score in zip(chunks, scores, strict=True)
    ]


class Retriever(Protocol):
    """Anything that ranks chunks for a query.

    ``default_min_score`` is the evidence-gate threshold on
    ``RetrievedChunk.relevance`` that suits this retriever's similarity scale.
    ``where`` restricts results to chunks whose metadata matches (see ``MetadataFilter``).
    """

    name: str
    default_min_score: float

    @property
    def chunks(self) -> list[DocumentChunk]: ...

    def search(
        self, query: str, *, top_k: int = 4, where: MetadataFilter | None = None
    ) -> list[RetrievedChunk]: ...


def rank(
    chunks: Sequence[DocumentChunk],
    scores: Sequence[float],
    *,
    top_k: int,
    relevance: Sequence[float] | None = None,
    where: MetadataFilter | None = None,
) -> list[RetrievedChunk]:
    """Return the ``top_k`` chunks with a positive score, best first (stable on ties)."""
    if top_k <= 0:
        return []
    scores = apply_filter(chunks, scores, where)
    order = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)
    return [
        RetrievedChunk(
            chunk=chunks[index],
            score=float(scores[index]),
            relevance=None if relevance is None else float(relevance[index]),
        )
        for index in order[:top_k]
        if scores[index] > 0
    ]


def save_chunks(chunks: Sequence[DocumentChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunk_file(path: Path) -> list[DocumentChunk]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DocumentChunk(**item) for item in payload]
