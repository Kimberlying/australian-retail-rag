from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..models import DocumentChunk, RetrievedChunk
from .base import rank
from .embeddings import Embedder, Vectors

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Cosine similarity over normalised embeddings, held in memory.

    An exact scan is the right choice at this corpus size (thousands of chunks
    score in well under a millisecond). The same interface can be backed by an
    ANN index (pgvector HNSW, Qdrant) when the corpus grows.

    Embeddings are cached next to the index and reused only when the model and
    the exact chunk ids match, so a re-chunk or model change can never serve
    stale vectors.
    """

    name = "dense"
    # Calibrated on the golden set (bge-small-en-v1.5): off-topic questions top out
    # at 0.49 and answerable ones start at 0.56. Re-check the sweep in the eval
    # report whenever the model, corpus, or golden set changes.
    default_min_score = 0.52

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        embedder: Embedder,
        *,
        cache_path: Path | None = None,
    ):
        self.chunks = list(chunks)
        self.embedder = embedder
        self._matrix = self._load_or_embed(cache_path)

    def _load_or_embed(self, cache_path: Path | None) -> Vectors:
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                if (
                    str(cached["model"]) == self.embedder.model_name
                    and list(cached["chunk_ids"]) == chunk_ids
                ):
                    return np.asarray(cached["vectors"], dtype=np.float32)
            logger.info("embedding cache is stale; re-embedding", extra={"fields": {}})

        if not self.chunks:
            return np.zeros((0, 0), dtype=np.float32)
        matrix = self.embedder.embed_documents([chunk.text for chunk in self.chunks])
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                model=np.array(self.embedder.model_name),
                chunk_ids=np.array(chunk_ids),
                vectors=matrix,
            )
        return matrix

    def score_all(self, query: str) -> list[float]:
        if not self.chunks:
            return []
        similarities = self._matrix @ self.embedder.embed_query(query)
        return [float(value) for value in np.clip(similarities, 0.0, 1.0)]

    def search(self, query: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        scores = self.score_all(query)
        return rank(self.chunks, scores, top_k=top_k, relevance=scores)
