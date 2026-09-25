from __future__ import annotations

from pathlib import Path

import pytest

from retail_rag.chunking import chunk_text
from retail_rag.ingest import build_index, load_chunks
from retail_rag.models import DocumentChunk
from retail_rag.retrieval import TfidfRetriever, tokenize


class TestChunking:
    def test_splits_long_text_into_multiple_non_empty_chunks(self) -> None:
        chunks = chunk_text(
            "one two three four five six seven eight nine ten", chunk_size=20, overlap=5
        )
        assert len(chunks) > 1
        assert all(chunks)

    def test_chunks_respect_size_limit(self) -> None:
        text = " ".join(f"word{i}" for i in range(500))
        assert all(len(chunk) <= 200 for chunk in chunk_text(text, chunk_size=200, overlap=40))

    def test_consecutive_chunks_overlap(self) -> None:
        text = " ".join(f"w{i}" for i in range(200))
        first, second = chunk_text(text, chunk_size=100, overlap=30)[:2]
        assert first.split()[-1] in second.split()

    def test_prefers_paragraph_boundaries(self) -> None:
        text = ("a" * 60) + "\n\n" + ("b" * 60)
        assert chunk_text(text, chunk_size=100, overlap=10)[0] == "a" * 60

    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_text("   \n\n  ") == []

    @pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
    def test_rejects_invalid_parameters(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError, match="must be"):
            chunk_text("text", chunk_size=size, overlap=overlap)


class TestRetrieval:
    def test_tokenize_lowercases_and_keeps_hyphenated_terms(self) -> None:
        assert tokenize("Click-and-Collect FY25") == ["click-and-collect", "fy25"]

    def test_returns_most_relevant_source_first(self, toy_chunks: list[DocumentChunk]) -> None:
        result = TfidfRetriever(toy_chunks).search("ecommerce sales growth")
        assert result[0].chunk.source == "sales.md"

    def test_scores_are_sorted_and_bounded(self, toy_chunks: list[DocumentChunk]) -> None:
        scores = [
            item.score for item in TfidfRetriever(toy_chunks).search("sales returns products")
        ]
        assert scores == sorted(scores, reverse=True)
        assert all(0 < score <= 1 for score in scores)

    def test_no_vocabulary_overlap_returns_nothing(self, toy_chunks: list[DocumentChunk]) -> None:
        assert TfidfRetriever(toy_chunks).search("zzz qqq") == []

    def test_non_positive_top_k_returns_nothing(self, toy_chunks: list[DocumentChunk]) -> None:
        assert TfidfRetriever(toy_chunks).search("sales", top_k=0) == []

    def test_save_load_round_trip(self, toy_chunks: list[DocumentChunk], tmp_path: Path) -> None:
        path = tmp_path / "nested" / "index.json"
        TfidfRetriever(toy_chunks).save(path)
        loaded = TfidfRetriever.load(path)
        assert loaded.chunks == toy_chunks
        assert loaded.search("human review")[0].chunk.chunk_id == "b"


class TestIngest:
    def test_loads_every_document_except_readme(self, docs_dir: Path) -> None:
        sources = {chunk.source for chunk in load_chunks(docs_dir)}
        assert "README.md" not in sources
        assert "coles_fy25_public_snapshot.md" in sources
        assert len(sources) == len(list(docs_dir.glob("*.md"))) - 1

    def test_chunk_ids_are_unique_and_deterministic(self, docs_dir: Path) -> None:
        first = [chunk.chunk_id for chunk in load_chunks(docs_dir)]
        assert len(first) == len(set(first))
        assert first == [chunk.chunk_id for chunk in load_chunks(docs_dir)]

    def test_build_index_persists_file(self, docs_dir: Path, tmp_path: Path) -> None:
        index = tmp_path / "index.json"
        retriever = build_index(docs_dir, index)
        assert index.exists()
        assert len(TfidfRetriever.load(index).chunks) == len(retriever.chunks)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_chunks(tmp_path / "missing")

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="No supported documents"):
            build_index(tmp_path, tmp_path / "index.json")
