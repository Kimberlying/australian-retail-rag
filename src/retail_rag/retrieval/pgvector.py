"""Dense retrieval backed by Postgres + pgvector (HNSW, cosine distance).

Same interface as the in-memory ``DenseRetriever``, so it slots into hybrid
search and reranking unchanged. Chunks are synced incrementally: only chunks
whose id is new (or whose embedding model changed) are embedded, and chunks
that no longer exist are deleted. Metadata filters run inside the SQL query, so
an HNSW scan never spends its candidate budget on excluded chunks.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from ..models import DocumentChunk, RetrievedChunk
from .base import MetadataFilter, rank

if TYPE_CHECKING:
    from .embeddings import Embedder, Vectors

logger = logging.getLogger(__name__)


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.7g}" for value in vector) + "]"


class PgVectorRetriever:
    name = "dense"
    # Same model and cosine scale as DenseRetriever, so the same calibrated gate.
    default_min_score = 0.575

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        embedder: Embedder,
        *,
        dsn: str,
        table: str = "rag_chunks",
        ef_search: int = 100,
    ):
        try:
            import psycopg  # noqa: PLC0415 - optional dependency
            from psycopg import sql  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "pgvector needs the 'pgvector' extra: uv sync --extra pgvector"
            ) from exc

        self.chunks = list(chunks)
        self.embedder = embedder
        self._sql = sql
        self._table_name = table
        self._table = sql.Identifier(table)
        self._index_name = sql.Identifier(f"{table}_embedding_hnsw")
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self._lock = threading.Lock()  # one connection shared by the API's worker threads
        self._conn = psycopg.connect(dsn, autocommit=True)
        # Candidates the HNSW graph walk keeps; higher = better recall, slower queries.
        self._conn.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(ef_search)))
        self._last_query: tuple[str, str] | None = None
        self.embedded_on_sync = self._sync()

    # ------------------------------------------------------------------ schema & sync

    def _sync(self) -> int:
        """Bring the table in line with ``self.chunks``; return how many chunks were embedded."""
        sql = self._sql
        dim = len(self.embedder.embed_query("dimension probe"))
        with self._lock, self._conn.transaction():
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            existing = self._conn.execute(
                "SELECT atttypmod FROM pg_attribute"
                " WHERE attrelid = to_regclass(%s) AND attname = 'embedding'",
                (self._table_name,),
            ).fetchone()
            if existing is not None and existing[0] != dim:
                # A model with a different dimension cannot share the column: rebuild.
                self._conn.execute(sql.SQL("DROP TABLE {}").format(self._table))
            self._conn.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} (chunk_id text PRIMARY KEY,"
                    " source text NOT NULL, text text NOT NULL, metadata jsonb NOT NULL,"
                    " model text NOT NULL, embedding vector({}) NOT NULL)"
                ).format(self._table, sql.Literal(dim))
            )
            self._conn.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops)"
                ).format(self._index_name, self._table)
            )
            ids = list(self._by_id)
            self._conn.execute(
                sql.SQL("DELETE FROM {} WHERE NOT (chunk_id = ANY(%s)) OR model <> %s").format(
                    self._table
                ),
                (ids, self.embedder.model_name),
            )
            present = {
                row[0]
                for row in self._conn.execute(
                    sql.SQL("SELECT chunk_id FROM {}").format(self._table)
                ).fetchall()
            }
            missing = [chunk for chunk in self.chunks if chunk.chunk_id not in present]
            if missing:
                vectors: Vectors = self.embedder.embed_documents([c.text for c in missing])
                with self._conn.cursor() as cursor:
                    cursor.executemany(
                        sql.SQL(
                            "INSERT INTO {} (chunk_id, source, text, metadata, model, embedding)"
                            " VALUES (%s, %s, %s, %s, %s, %s::vector)"
                        ).format(self._table),
                        [
                            (
                                chunk.chunk_id,
                                chunk.source,
                                chunk.text,
                                json.dumps(chunk.metadata),
                                self.embedder.model_name,
                                _vector_literal(vector.tolist()),
                            )
                            for chunk, vector in zip(missing, vectors, strict=True)
                        ],
                    )
        logger.info(
            "pgvector sync",
            extra={"fields": {"chunks": len(self.chunks), "embedded": len(missing)}},
        )
        return len(missing)

    # ------------------------------------------------------------------ queries

    def _query_vector(self, query: str) -> str:
        cached = self._last_query
        if cached is not None and cached[0] == query:
            return cached[1]
        literal = _vector_literal(self.embedder.embed_query(query).tolist())
        self._last_query = (query, literal)
        return literal

    def _where_sql(self, where: MetadataFilter | None) -> tuple[Any, list[Any]]:
        sql = self._sql
        if not where:
            return sql.SQL("TRUE"), []
        clauses = []
        params: list[Any] = []
        for key, allowed in where.items():
            # Same rules as matches_filter(): scalar equality or list containment; a
            # missing key yields NULL, which only passes when None is allowed.
            values = [value for value in allowed if value is not None]
            clause = sql.SQL(
                "(metadata -> %s = ANY(%s::jsonb[]) OR metadata -> %s @> ANY(%s::jsonb[])"
            )
            params.extend(
                [
                    key,
                    [json.dumps(value) for value in values],
                    key,
                    [json.dumps([value]) for value in values],
                ]
            )
            if None in allowed:
                clause = sql.SQL("{} OR NOT metadata ? %s").format(clause)
                params.append(key)
            clauses.append(sql.SQL("{})").format(clause))
        return sql.SQL(" AND ").join(clauses), params

    def search(
        self, query: str, *, top_k: int = 4, where: MetadataFilter | None = None
    ) -> list[RetrievedChunk]:
        if top_k <= 0 or not self.chunks:
            return []
        vector = self._query_vector(query)
        condition, params = self._where_sql(where)
        statement = self._sql.SQL(
            "SELECT chunk_id, 1 - (embedding <=> %s::vector) FROM {} WHERE {}"
            " ORDER BY embedding <=> %s::vector LIMIT %s"
        ).format(self._table, condition)
        with self._lock:
            rows = self._conn.execute(statement, [vector, *params, vector, top_k]).fetchall()
        found = [self._by_id[chunk_id] for chunk_id, _ in rows]
        scores = [min(max(float(similarity), 0.0), 1.0) for _, similarity in rows]
        return rank(found, scores, top_k=top_k, relevance=scores)

    def similarities(self, query: str, chunk_ids: Iterable[str]) -> dict[str, float]:
        ids = list(chunk_ids)
        if not ids:
            return {}
        statement = self._sql.SQL(
            "SELECT chunk_id, 1 - (embedding <=> %s::vector) FROM {} WHERE chunk_id = ANY(%s)"
        ).format(self._table)
        with self._lock:
            rows = self._conn.execute(statement, [self._query_vector(query), ids]).fetchall()
        return {chunk_id: min(max(float(value), 0.0), 1.0) for chunk_id, value in rows}

    def close(self) -> None:
        self._conn.close()
