from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Retriever, load_chunk_file
from .bm25 import BM25Retriever
from .tfidf import TfidfRetriever

if TYPE_CHECKING:
    from ..config import RetrieverKind, Settings
    from ..models import DocumentChunk
    from .embeddings import Embedder


def embeddings_cache_path(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.stem}.embeddings.npz")


def default_embedder(settings: Settings) -> Embedder:
    from .embeddings import FastEmbedEmbedder  # noqa: PLC0415 - loads ONNX runtime lazily

    return FastEmbedEmbedder(
        settings.embedding_model,
        model_path=settings.embedding_model_path,
        cache_dir=settings.embedding_cache_dir,
    )


def create_retriever(
    kind: RetrieverKind,
    chunks: Sequence[DocumentChunk],
    settings: Settings,
    *,
    cache_path: Path | None = None,
    embedder: Embedder | None = None,
) -> Retriever:
    if kind == "tfidf":
        return TfidfRetriever(chunks)
    if kind == "bm25":
        return BM25Retriever(chunks)

    from .dense import DenseRetriever  # noqa: PLC0415
    from .hybrid import HybridRetriever  # noqa: PLC0415

    dense = DenseRetriever(chunks, embedder or default_embedder(settings), cache_path=cache_path)
    if kind == "dense":
        return dense
    return HybridRetriever(BM25Retriever(chunks), dense)


def load_retriever(
    index_path: Path, settings: Settings, *, embedder: Embedder | None = None
) -> Retriever:
    """Load persisted chunks and build the configured retriever over them."""
    return create_retriever(
        settings.retriever,
        load_chunk_file(index_path),
        settings,
        cache_path=embeddings_cache_path(index_path),
        embedder=embedder,
    )
