from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from retail_rag.config import Settings
from retail_rag.evaluation import metrics
from retail_rag.ingest import build_index
from retail_rag.models import DocumentChunk
from retail_rag.pipeline import RAGPipeline
from retail_rag.retrieval import (
    BM25Retriever,
    HybridRetriever,
    TfidfRetriever,
    create_retriever,
    embeddings_cache_path,
    load_retriever,
    reciprocal_rank_fusion,
)
from retail_rag.retrieval.base import matches_filter
from retail_rag.retrieval.dense import DenseRetriever
from retail_rag.retrieval.rerank import RerankingRetriever

from .conftest import DOCS_DIR, FakeEmbedder


@pytest.fixture
def chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk("frozen", "cold.md", "Frozen products must be held at or below -18C."),
        DocumentChunk("ebit", "coles.md", "Coles FY25 group EBIT was A$2.1 billion."),
        DocumentChunk("theft", "ops.md", "Staff must never confront a suspected shoplifter."),
        DocumentChunk(
            "long",
            "long.md",
            "Frozen frozen. " + "Unrelated filler text about store layouts and signage. " * 20,
        ),
    ]


class TestBM25:
    def test_ranks_exact_term_match_first(self, chunks: list[DocumentChunk]) -> None:
        assert BM25Retriever(chunks).search("Coles EBIT")[0].chunk.chunk_id == "ebit"

    def test_stopword_only_query_matches_nothing(self, chunks: list[DocumentChunk]) -> None:
        assert BM25Retriever(chunks).search("what is the of") == []

    def test_length_normalisation_prefers_focused_chunk(self, chunks: list[DocumentChunk]) -> None:
        # "long" has frozen x2 but is ~20x longer; BM25 should still rank "frozen" first.
        assert BM25Retriever(chunks).search("frozen")[0].chunk.chunk_id == "frozen"

    def test_has_no_calibrated_relevance(self, chunks: list[DocumentChunk]) -> None:
        assert all(item.relevance is None for item in BM25Retriever(chunks).search("frozen"))


class TestReciprocalRankFusion:
    def test_matches_hand_computation(self) -> None:
        fused = reciprocal_rank_fusion([[1, 2], [2, 3]], k=60)
        assert fused[2] == pytest.approx(1 / 62 + 1 / 61)
        assert fused[1] == pytest.approx(1 / 61)
        assert fused[3] == pytest.approx(1 / 62)
        assert max(fused, key=fused.__getitem__) == 2  # agreed on by both lists

    def test_weights(self) -> None:
        fused = reciprocal_rank_fusion([[1], [2]], k=0, weights=[2.0, 1.0])
        assert fused == {1: 2.0, 2: 1.0}


class TestDense:
    def test_paraphrase_match_via_embeddings(self, chunks: list[DocumentChunk]) -> None:
        embedder = FakeEmbedder(synonyms={"freezer": "frozen", "stealing": "shoplifter"})
        dense = DenseRetriever(chunks, embedder)
        assert dense.search("freezer section")[0].chunk.chunk_id == "frozen"
        assert dense.search("someone stealing")[0].chunk.chunk_id == "theft"
        # A lexical retriever cannot bridge the vocabulary gap.
        assert BM25Retriever(chunks).search("stealing") == []

    def test_relevance_is_cosine_in_unit_range(
        self, chunks: list[DocumentChunk], fake_embedder: FakeEmbedder
    ) -> None:
        results = DenseRetriever(chunks, fake_embedder).search("frozen products", top_k=4)
        assert all(item.relevance == item.score for item in results)
        assert all(0 < item.score <= 1 for item in results)

    def test_embedding_cache_is_reused_then_invalidated(
        self, chunks: list[DocumentChunk], tmp_path: Path
    ) -> None:
        cache = tmp_path / "index.embeddings.npz"
        embedder = FakeEmbedder()
        DenseRetriever(chunks, embedder, cache_path=cache)
        DenseRetriever(chunks, embedder, cache_path=cache)
        assert embedder.document_calls == 1  # second load hit the cache

        DenseRetriever(chunks[:2], embedder, cache_path=cache)  # chunk ids changed
        assert embedder.document_calls == 2
        DenseRetriever(chunks[:2], FakeEmbedder(dim=32), cache_path=cache)  # model changed
        assert embedder.document_calls == 2  # the new embedder did the work instead

    def test_empty_corpus(self, fake_embedder: FakeEmbedder) -> None:
        assert DenseRetriever([], fake_embedder).search("anything") == []


class TestHybrid:
    def test_keeps_candidates_found_by_either_retriever(self, chunks: list[DocumentChunk]) -> None:
        embedder = FakeEmbedder(synonyms={"stealing": "shoplifter"})
        hybrid = HybridRetriever(BM25Retriever(chunks), DenseRetriever(chunks, embedder))
        assert hybrid.search("someone stealing")[0].chunk.chunk_id == "theft"  # dense only
        assert hybrid.search("Coles EBIT")[0].chunk.chunk_id == "ebit"  # both agree

    def test_relevance_comes_from_dense_similarity(
        self, chunks: list[DocumentChunk], fake_embedder: FakeEmbedder
    ) -> None:
        dense = DenseRetriever(chunks, fake_embedder)
        hybrid = HybridRetriever(BM25Retriever(chunks), dense)
        dense_scores = dict(
            zip([c.chunk_id for c in chunks], dense.score_all("frozen"), strict=True)
        )
        for item in hybrid.search("frozen", top_k=4):
            assert item.relevance == pytest.approx(dense_scores[item.chunk.chunk_id])

    def test_rejects_mismatched_indexes(
        self, chunks: list[DocumentChunk], fake_embedder: FakeEmbedder
    ) -> None:
        with pytest.raises(ValueError, match="same chunks"):
            HybridRetriever(BM25Retriever(chunks[:2]), DenseRetriever(chunks, fake_embedder))


class TestFactoryAndPersistence:
    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("tfidf", TfidfRetriever),
            ("bm25", BM25Retriever),
            ("dense", DenseRetriever),
            ("hybrid", HybridRetriever),
        ],
    )
    def test_create_retriever(
        self,
        kind: str,
        expected: type,
        chunks: list[DocumentChunk],
        settings: Settings,
        fake_embedder: FakeEmbedder,
    ) -> None:
        retriever = create_retriever(kind, chunks, settings, embedder=fake_embedder)  # type: ignore[arg-type]
        assert isinstance(retriever, expected)
        assert retriever.name == kind

    def test_build_then_load_hybrid_reuses_cached_embeddings(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        hybrid_settings = settings.model_copy(update={"retriever": "hybrid"})
        index = tmp_path / "index.json"
        embedder = FakeEmbedder()
        build_index(DOCS_DIR, index, hybrid_settings, embedder=embedder)
        assert embeddings_cache_path(index).exists()

        loaded = load_retriever(index, hybrid_settings, embedder=embedder)
        assert isinstance(loaded, HybridRetriever)
        assert embedder.document_calls == 1  # loading did not re-embed


class TestEvidenceGate:
    def test_refuses_when_best_relevance_is_below_threshold(
        self, chunks: list[DocumentChunk], settings: Settings, fake_embedder: FakeEmbedder
    ) -> None:
        strict = settings.model_copy(update={"min_score": 0.99})
        result = RAGPipeline(DenseRetriever(chunks, fake_embedder), strict).ask("frozen goods")
        assert result.refused
        assert result.citations == []
        assert result.evidence_score is not None
        assert result.evidence_score < 0.99

    def test_keeps_every_chunk_once_the_gate_passes(
        self, chunks: list[DocumentChunk], settings: Settings, fake_embedder: FakeEmbedder
    ) -> None:
        pipeline = RAGPipeline(DenseRetriever(chunks, fake_embedder), settings)
        result = pipeline.ask("frozen products", top_k=4)
        relevances = [citation["relevance"] for citation in result.citations]
        assert not result.refused
        assert max(relevances) >= pipeline.min_score
        # The gate is answer-level: a weaker supporting chunk is not filtered out.
        assert min(relevances) < pipeline.min_score

    def test_uses_retriever_default_when_unset(
        self, chunks: list[DocumentChunk], settings: Settings, fake_embedder: FakeEmbedder
    ) -> None:
        dense = DenseRetriever(chunks, fake_embedder)
        assert RAGPipeline(dense, settings).min_score == dense.default_min_score
        explicit = settings.model_copy(update={"min_score": 0.3})
        assert RAGPipeline(dense, explicit).min_score == 0.3

    def test_bm25_has_no_gate(self, chunks: list[DocumentChunk], settings: Settings) -> None:
        strict = settings.model_copy(update={"min_score": 0.99})
        result = RAGPipeline(BM25Retriever(chunks), strict).ask("group EBIT")
        assert not result.refused
        assert result.evidence_score is None


def test_refusal_sweep() -> None:
    rows = metrics.refusal_sweep([0.6, 0.8], [0.3, 0.7], [0.5, 0.75])
    assert rows[0] == {"threshold": 0.5, "refusal_accuracy": 0.5, "false_refusal_rate": 0.0}
    assert rows[1] == {"threshold": 0.75, "refusal_accuracy": 1.0, "false_refusal_rate": 0.5}


@pytest.mark.skipif(
    not os.environ.get("RAG_EMBEDDING_MODEL_PATH") and not os.environ.get("RAG_TEST_REAL_MODEL"),
    reason="set RAG_EMBEDDING_MODEL_PATH or RAG_TEST_REAL_MODEL=1 to run the real embedding model",
)
def test_real_model_bridges_paraphrase(settings: Settings) -> None:
    """Integration check against bge-small: 'freezer' should land on the frozen rule."""
    from retail_rag.retrieval.factory import default_embedder

    embedder = default_embedder(settings)
    corpus = [
        DocumentChunk("frozen", "a.md", "Frozen products must be held at or below -18°C."),
        DocumentChunk("ebit", "b.md", "Coles FY25 group EBIT was A$2.1 billion."),
    ]
    top = DenseRetriever(corpus, embedder).search("How cold does the freezer section need to be?")
    assert top[0].chunk.chunk_id == "frozen"
    assert top[0].score > 0.7


class FakeReranker:
    """Scores by how many of ``preferred`` words a text contains (a stand-in cross-encoder)."""

    model_name = "fake-cross-encoder"

    def __init__(self, preferred: set[str]):
        self.preferred = preferred
        self.calls = 0

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls += 1
        return [float(sum(word in text.lower() for word in self.preferred)) for text in texts]


class TestMetadataFilter:
    @pytest.fixture
    def dated(self) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                "fy25",
                "coles.md",
                "Coles revenue was A$44.4 billion.",
                {"company": "Coles", "fiscal_year": 2025},
            ),
            DocumentChunk(
                "fy24",
                "coles24.md",
                "Coles revenue was A$43.6 billion.",
                {"company": "Coles", "fiscal_year": 2024},
            ),
            DocumentChunk("undated", "ops.md", "Revenue targets are reviewed monthly.", {}),
        ]

    def test_matches_filter_requires_key_and_allowed_value(
        self, dated: list[DocumentChunk]
    ) -> None:
        where = {"fiscal_year": [2025]}
        assert [matches_filter(chunk, where) for chunk in dated] == [True, False, False]
        undated_ok = {"fiscal_year": [2025, None]}
        assert [matches_filter(chunk, undated_ok) for chunk in dated] == [True, False, True]
        assert all(matches_filter(chunk, None) for chunk in dated)

    @pytest.mark.parametrize("kind", ["tfidf", "bm25", "dense", "hybrid"])
    def test_every_retriever_honours_where(
        self, kind: str, dated: list[DocumentChunk], settings: Settings
    ) -> None:
        retriever = create_retriever(kind, dated, settings, embedder=FakeEmbedder())  # type: ignore[arg-type]
        results = retriever.search("Coles revenue", top_k=3, where={"fiscal_year": [2024]})
        assert [item.chunk.chunk_id for item in results] == ["fy24"]
        assert retriever.search("Coles revenue", where={"company": ["Woolworths"]}) == []


class TestReranker:
    def test_reorders_first_stage_candidates(self, chunks: list[DocumentChunk]) -> None:
        base = BM25Retriever(chunks)
        assert base.search("frozen")[0].chunk.chunk_id == "frozen"
        reranked = RerankingRetriever(base, FakeReranker({"filler"}), candidates=10)
        results = reranked.search("frozen", top_k=2)
        assert results[0].chunk.chunk_id == "long"  # the cross-encoder preferred it
        assert reranked.name == "bm25+rerank"

    def test_relevance_passes_through_from_first_stage(
        self, chunks: list[DocumentChunk], fake_embedder: FakeEmbedder
    ) -> None:
        dense = DenseRetriever(chunks, fake_embedder)
        reranked = RerankingRetriever(dense, FakeReranker({"shoplifter"}))
        first_stage = {
            item.chunk.chunk_id: item.relevance for item in dense.search("frozen", top_k=20)
        }
        for item in reranked.search("frozen", top_k=3):
            assert item.relevance == first_stage[item.chunk.chunk_id]
        assert reranked.default_min_score == dense.default_min_score

    def test_skips_model_call_for_single_candidate(self, chunks: list[DocumentChunk]) -> None:
        reranker = FakeReranker(set())
        results = RerankingRetriever(BM25Retriever(chunks), reranker).search("EBIT")
        assert [item.chunk.chunk_id for item in results] == ["ebit"]
        assert reranker.calls == 0
        assert RerankingRetriever(BM25Retriever(chunks), reranker).search("x", top_k=0) == []

    def test_rejects_non_positive_candidates(self, chunks: list[DocumentChunk]) -> None:
        with pytest.raises(ValueError, match="candidates"):
            RerankingRetriever(BM25Retriever(chunks), FakeReranker(set()), candidates=0)

    def test_factory_wraps_when_configured(
        self, chunks: list[DocumentChunk], settings: Settings
    ) -> None:
        configured = settings.model_copy(update={"reranker": "cross-encoder"})
        retriever = create_retriever("bm25", chunks, configured, reranker=FakeReranker(set()))
        assert isinstance(retriever, RerankingRetriever)
        assert retriever.chunks == chunks
        assert not isinstance(create_retriever("bm25", chunks, settings), RerankingRetriever)


PG_URL = os.environ.get("RAG_TEST_DATABASE_URL")


@pytest.mark.skipif(not PG_URL, reason="set RAG_TEST_DATABASE_URL to a Postgres with pgvector")
class TestPgVector:
    @pytest.fixture
    def corpus(self) -> list[DocumentChunk]:
        from retail_rag.ingest import load_chunks

        return load_chunks(DOCS_DIR)

    def _pg(self, chunks: list[DocumentChunk], embedder: FakeEmbedder) -> Any:
        from retail_rag.retrieval.pgvector import PgVectorRetriever

        assert PG_URL is not None
        return PgVectorRetriever(chunks, embedder, dsn=PG_URL, table="rag_chunks_test")

    def test_matches_in_memory_dense_search(self, corpus: list[DocumentChunk]) -> None:
        embedder = FakeEmbedder()
        memory, pg = DenseRetriever(corpus, embedder), self._pg(corpus, embedder)
        for query in ["frozen products temperature", "Coles revenue", "loyalty points expiry"]:
            expected = [(r.chunk.chunk_id, round(r.score, 4)) for r in memory.search(query)]
            assert [(r.chunk.chunk_id, round(r.score, 4)) for r in pg.search(query)] == expected
        hybrid = HybridRetriever(BM25Retriever(corpus), pg)
        reference = HybridRetriever(BM25Retriever(corpus), memory)
        assert [r.chunk.chunk_id for r in hybrid.search("recall notice")] == [
            r.chunk.chunk_id for r in reference.search("recall notice")
        ]
        pg.close()

    def test_sync_is_incremental(self, corpus: list[DocumentChunk]) -> None:
        embedder = FakeEmbedder()
        self._pg(corpus[:-1], embedder).close()
        again = self._pg(corpus, embedder)
        assert again.embedded_on_sync == 1  # only the new chunk
        again.close()
        unchanged = self._pg(corpus, FakeEmbedder())
        assert unchanged.embedded_on_sync == 0
        unchanged.close()
        new_model = self._pg(corpus, FakeEmbedder(dim=32))  # new dimension: table rebuilt
        assert new_model.embedded_on_sync == len(corpus)
        assert new_model.search("frozen products")
        new_model.close()

    def test_filters_run_in_sql(self, corpus: list[DocumentChunk]) -> None:
        pg = self._pg(corpus, FakeEmbedder())
        coles = pg.search("revenue", top_k=10, where={"company": ["Coles"]})
        assert coles
        assert {item.chunk.source for item in coles} == {"coles_fy25_public_snapshot.md"}
        assert pg.search("revenue", where={"company": ["Woolworths"]}) == []
        undated_ok = pg.search("revenue", top_k=50, where={"fiscal_year": [2024, None]})
        memory = DenseRetriever(corpus, FakeEmbedder())
        expected = memory.search("revenue", top_k=50, where={"fiscal_year": [2024, None]})
        assert {r.chunk.chunk_id for r in undated_ok} == {r.chunk.chunk_id for r in expected}
        assert "coles_fy25_public_snapshot.md" not in {r.chunk.source for r in undated_ok}
        pg.close()
