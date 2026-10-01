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
    from .rerank import Reranker


def embeddings_cache_path(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.stem}.embeddings.npz")


def default_embedder(settings: Settings) -> Embedder:
    from .embeddings import FastEmbedEmbedder  # noqa: PLC0415 - loads ONNX runtime lazily

    return FastEmbedEmbedder(
        settings.embedding_model,
        model_path=settings.embedding_model_path,
        cache_dir=settings.embedding_cache_dir,
    )


def default_reranker(settings: Settings) -> Reranker:
    from .rerank import FastEmbedReranker  # noqa: PLC0415 - loads ONNX runtime lazily

    return FastEmbedReranker(
        settings.reranker_model,
        model_path=settings.reranker_model_path,
        cache_dir=settings.embedding_cache_dir,
    )


def _first_stage(
    kind: RetrieverKind,
    chunks: Sequence[DocumentChunk],
    settings: Settings,
    *,
    cache_path: Path | None,
    embedder: Embedder | None,
) -> Retriever:
    if kind == "tfidf":
        return TfidfRetriever(chunks)
    if kind == "bm25":
        return BM25Retriever(chunks)

    from .hybrid import DenseSearch, HybridRetriever  # noqa: PLC0415

    embedder = embedder or default_embedder(settings)
    dense: DenseSearch
    if settings.vector_store == "pgvector":
        from .pgvector import PgVectorRetriever  # noqa: PLC0415 - needs the 'pgvector' extra

        if settings.database_url is None:
            raise ValueError("RAG_VECTOR_STORE=pgvector needs RAG_DATABASE_URL")
        dense = PgVectorRetriever(chunks, embedder, dsn=settings.database_url.get_secret_value())
    else:
        from .dense import DenseRetriever  # noqa: PLC0415

        dense = DenseRetriever(chunks, embedder, cache_path=cache_path)
    if kind == "dense":
        return dense
    return HybridRetriever(BM25Retriever(chunks), dense)


def create_retriever(
    kind: RetrieverKind,
    chunks: Sequence[DocumentChunk],
    settings: Settings,
    *,
    cache_path: Path | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> Retriever:
    """Build the first-stage retriever and, if configured, wrap it in a cross-encoder."""
    retriever = _first_stage(kind, chunks, settings, cache_path=cache_path, embedder=embedder)
    if reranker is None and settings.reranker == "none":
        return retriever

    from .rerank import RerankingRetriever  # noqa: PLC0415

    return RerankingRetriever(
        retriever, reranker or default_reranker(settings), candidates=settings.rerank_candidates
    )


def load_retriever(
    index_path: Path,
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> Retriever:
    """Load persisted chunks and build the configured retriever over them."""
    return create_retriever(
        settings.retriever,
        load_chunk_file(index_path),
        settings,
        cache_path=embeddings_cache_path(index_path),
        embedder=embedder,
        reranker=reranker,
    )
