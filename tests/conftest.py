from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from retail_rag.config import REPO_ROOT, Settings, get_settings
from retail_rag.ingest import load_chunks
from retail_rag.models import DocumentChunk
from retail_rag.pipeline import RAGPipeline
from retail_rag.retrieval import TfidfRetriever

DOCS_DIR = REPO_ROOT / "data" / "documents"
GOLDEN_SET = REPO_ROOT / "evals" / "golden_set.jsonl"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Never let a developer's real key or .env leak into unit tests."""
    for name in ("ANTHROPIC_API_KEY", "RAG_INDEX_PATH", "RAG_DOCS_DIR", "RAG_MIN_SCORE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(REPO_ROOT / "tests")  # no .env in this directory
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def toy_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk("a", "sales.md", "normalised ecommerce sales growth was 23.3 percent"),
        DocumentChunk("b", "policy.md", "unknown returns require human review"),
        DocumentChunk("c", "cold.md", "chilled products must be held between 0 and 5 degrees"),
    ]


@pytest.fixture
def corpus_pipeline(settings: Settings) -> RAGPipeline:
    return RAGPipeline(TfidfRetriever(load_chunks(DOCS_DIR)), settings)


@pytest.fixture
def docs_dir() -> Path:
    return DOCS_DIR
