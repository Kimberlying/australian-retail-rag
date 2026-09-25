from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import anthropic
import pytest
from pydantic import SecretStr

from retail_rag import generation
from retail_rag.config import Settings
from retail_rag.generation import REFUSAL_TEXT, build_context, generate_with_claude, is_refusal
from retail_rag.models import DocumentChunk, RetrievedChunk
from retail_rag.pipeline import RAGPipeline
from retail_rag.retrieval import TfidfRetriever


class FakeAnthropic:
    """Stands in for ``anthropic.Anthropic``; records the request it receives."""

    last_request: ClassVar[dict[str, Any]] = {}

    def __init__(self, *, reply: str = "", stop_reason: str = "end_turn", **_: Any) -> None:
        self._reply = reply
        self._stop_reason = stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        FakeAnthropic.last_request = kwargs
        return SimpleNamespace(
            model=kwargs["model"],
            stop_reason=self._stop_reason,
            content=[SimpleNamespace(type="text", text=self._reply)],
            usage=SimpleNamespace(input_tokens=120, output_tokens=30),
        )


def _install_fake(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
    monkeypatch.setattr(anthropic, "Anthropic", lambda **client_kwargs: FakeAnthropic(**kwargs))


@pytest.fixture
def llm_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"anthropic_api_key": SecretStr("test-key")})


@pytest.fixture
def evidence() -> list[RetrievedChunk]:
    chunk = DocumentChunk("abc", "sales.md", "Growth was 23.3% <b>strong</b>")
    return [RetrievedChunk(chunk=chunk, score=0.8)]


class TestPromptConstruction:
    def test_context_wraps_documents_in_xml_and_escapes_markup(
        self, evidence: list[RetrievedChunk]
    ) -> None:
        context = build_context(evidence)
        assert context.startswith("<documents>")
        assert 'source="sales.md"' in context
        assert "&lt;b&gt;strong&lt;/b&gt;" in context  # document text cannot inject tags

    def test_empty_context(self) -> None:
        assert build_context([]) == "<documents></documents>"

    def test_refusal_detection(self) -> None:
        assert is_refusal(f"Sorry. {REFUSAL_TEXT}")
        assert not is_refusal("Growth was 23.3%.")


class TestClaudeGeneration:
    def test_returns_none_without_api_key(
        self, settings: Settings, evidence: list[RetrievedChunk]
    ) -> None:
        assert generate_with_claude("q", evidence, settings) is None

    def test_sends_grounded_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
        llm_settings: Settings,
        evidence: list[RetrievedChunk],
    ) -> None:
        _install_fake(monkeypatch, reply="Growth was 23.3%.\nSources: sales.md")
        result = generate_with_claude("What was growth?", evidence, llm_settings)

        assert result is not None
        assert result.text.startswith("Growth was 23.3%")
        assert not result.refused
        assert result.usage == {"input_tokens": 120, "output_tokens": 30}
        request = FakeAnthropic.last_request
        assert request["model"] == llm_settings.anthropic_model
        assert request["system"] == generation.SYSTEM_PROMPT
        assert "<question>What was growth?</question>" in request["messages"][0]["content"]

    def test_model_refusal_text_is_flagged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        llm_settings: Settings,
        evidence: list[RetrievedChunk],
    ) -> None:
        _install_fake(monkeypatch, reply=REFUSAL_TEXT)
        result = generate_with_claude("Woolworths EBIT?", evidence, llm_settings)
        assert result is not None
        assert result.refused

    def test_safety_refusal_stop_reason_is_flagged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        llm_settings: Settings,
        evidence: list[RetrievedChunk],
    ) -> None:
        _install_fake(monkeypatch, stop_reason="refusal")
        result = generate_with_claude("q", evidence, llm_settings)
        assert result is not None
        assert result.refused
        assert result.text == REFUSAL_TEXT


class TestPipeline:
    def test_local_mode_answers_with_evidence_and_citations(
        self, corpus_pipeline: RAGPipeline
    ) -> None:
        result = corpus_pipeline.ask("What is the normalised eCommerce sales growth?")
        assert result.generated_by == "local"
        assert "23.3" in result.answer
        assert result.citations
        assert result.citations[0]["source"] == "coles_fy25_public_snapshot.md"
        assert set(result.latency_ms) == {"retrieval", "generation", "total"}

    def test_refuses_without_llm_call_when_nothing_clears_threshold(
        self, settings: Settings, toy_chunks: list[DocumentChunk]
    ) -> None:
        pipeline = RAGPipeline(TfidfRetriever(toy_chunks), settings)
        result = pipeline.ask("zzz qqq")
        assert result.refused
        assert result.answer == REFUSAL_TEXT
        assert result.citations == []

    def test_min_score_filters_weak_evidence(
        self, settings: Settings, toy_chunks: list[DocumentChunk]
    ) -> None:
        strict = settings.model_copy(update={"min_score": 0.99})
        assert RAGPipeline(TfidfRetriever(toy_chunks), strict).ask("sales").refused

    def test_uses_claude_when_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
        llm_settings: Settings,
        toy_chunks: list[DocumentChunk],
    ) -> None:
        _install_fake(monkeypatch, reply="Growth was 23.3%. Sources: sales.md")
        result = RAGPipeline(TfidfRetriever(toy_chunks), llm_settings).ask("sales growth")
        assert result.generated_by == "claude"
        assert result.usage["output_tokens"] == 30

    def test_falls_back_to_local_preview_when_claude_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        llm_settings: Settings,
        toy_chunks: list[DocumentChunk],
    ) -> None:
        def boom(*_: Any, **__: Any) -> None:
            raise ConnectionError("network down")

        monkeypatch.setattr("retail_rag.pipeline.generate_with_claude", boom)
        result = RAGPipeline(TfidfRetriever(toy_chunks), llm_settings).ask("sales growth")
        assert result.generated_by == "local_fallback"
        assert "23.3" in result.answer
