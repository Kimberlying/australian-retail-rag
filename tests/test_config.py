from __future__ import annotations

import pytest
from pydantic import ValidationError

from retail_rag.config import REPO_ROOT, Settings


def test_defaults_resolve_relative_to_repo_root() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.docs_dir == REPO_ROOT / "data" / "documents"
    assert settings.index_path.is_absolute()
    assert not settings.llm_enabled


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.top_k == 7
    assert settings.llm_enabled
    assert "sk-test" not in repr(settings)  # secrets never leak into logs


def test_blank_api_key_means_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
    assert not Settings(_env_file=None).llm_enabled  # type: ignore[call-arg]


def test_rejects_overlap_not_smaller_than_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", "100")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "100")
    with pytest.raises(ValidationError, match="RAG_CHUNK_OVERLAP"):
        Settings(_env_file=None)  # type: ignore[call-arg]
