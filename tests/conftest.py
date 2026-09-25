from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from retail_rag.config import REPO_ROOT, Settings, get_settings
from retail_rag.ingest import load_chunks
from retail_rag.models import DocumentChunk
from retail_rag.pipeline import RAGPipeline
from retail_rag.retrieval import TfidfRetriever, content_tokens
from retail_rag.retrieval.embeddings import Vectors, normalise

DOCS_DIR = REPO_ROOT / "data" / "documents"
GOLDEN_SET = REPO_ROOT / "evals" / "golden_set.jsonl"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Never let a developer's real key or .env leak into unit tests."""
    for name in ("ANTHROPIC_API_KEY", "RAG_INDEX_PATH", "RAG_DOCS_DIR", "RAG_MIN_SCORE"):
        monkeypatch.delenv(name, raising=False)
    # Unit tests default to the dependency-free retriever; dense/hybrid tests opt in
    # explicitly with FakeEmbedder so the suite never downloads a model.
    monkeypatch.setenv("RAG_RETRIEVER", "tfidf")
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


class FakeEmbedder:
    """Deterministic hashed bag-of-words embeddings: offline, fast, and stable.

    ``synonyms`` maps words onto a shared bucket so tests can express paraphrases.
    """

    def __init__(self, dim: int = 64, synonyms: dict[str, str] | None = None):
        self.model_name = f"fake-hash-{dim}"
        self.dim = dim
        self.synonyms = synonyms or {}
        self.document_calls = 0

    def _vector(self, text: str) -> Vectors:
        vector = np.zeros(self.dim, dtype=np.float32)
        for raw in content_tokens(text):
            token = self.synonyms.get(raw, raw)
            bucket = int(hashlib.md5(token.encode(), usedforsecurity=False).hexdigest(), 16)
            vector[bucket % self.dim] += 1.0
        return vector

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        self.document_calls += 1
        return normalise(np.stack([self._vector(text) for text in texts]))

    def embed_query(self, text: str) -> Vectors:
        return normalise(self._vector(text))


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
