"""Shared retriever contract, tokenisation, and chunk persistence."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

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


class Retriever(Protocol):
    """Anything that ranks chunks for a query.

    ``default_min_score`` is the evidence-gate threshold on
    ``RetrievedChunk.relevance`` that suits this retriever's similarity scale.
    """

    name: str
    chunks: list[DocumentChunk]
    default_min_score: float

    def search(self, query: str, *, top_k: int = 4) -> list[RetrievedChunk]: ...


def rank(
    chunks: Sequence[DocumentChunk],
    scores: Sequence[float],
    *,
    top_k: int,
    relevance: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """Return the ``top_k`` chunks with a positive score, best first (stable on ties)."""
    if top_k <= 0:
        return []
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
