from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from retail_rag import tracing
from retail_rag.config import Settings
from retail_rag.models import DocumentChunk
from retail_rag.pipeline import RAGPipeline
from retail_rag.retrieval import TfidfRetriever

from .fakes import ScriptedAnthropic, install, message, text_block

_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    # Route this module's tracer to an in-memory exporter without touching the
    # process-wide provider (which OpenTelemetry allows setting only once).
    monkeypatch.setattr(trace, "get_tracer", lambda *args, **kwargs: _PROVIDER.get_tracer("test"))
    _EXPORTER.clear()
    yield _EXPORTER


def test_span_is_noop_without_provider() -> None:
    with tracing.span("anything", key="value", skipped=None) as current:
        current.set_attribute("x", 1)  # must not raise


def test_query_produces_nested_spans_with_genai_attributes(
    monkeypatch: pytest.MonkeyPatch,
    spans: InMemorySpanExporter,
    settings: Settings,
    toy_chunks: list[DocumentChunk],
) -> None:
    llm = settings.model_copy(update={"anthropic_api_key": SecretStr("k")})
    install(monkeypatch, ScriptedAnthropic([message(text_block("23.3%"), cache_read=512)]))
    RAGPipeline(TfidfRetriever(toy_chunks), llm).ask("sales growth")

    finished = {span.name: span for span in spans.get_finished_spans()}
    assert {"rag.query", "rag.route", "rag.retrieve", "gen_ai.generate"} <= set(finished)
    root = finished["rag.query"]
    for child in ("rag.route", "rag.retrieve", "gen_ai.generate"):
        assert finished[child].parent is not None
        assert finished[child].parent.span_id == root.context.span_id
    generate = finished["gen_ai.generate"].attributes or {}
    assert generate["gen_ai.system"] == "anthropic"
    assert generate["gen_ai.usage.input_tokens"] == 100
    assert generate["gen_ai.usage.cache_read_input_tokens"] == 512
    assert (finished["rag.query"].attributes or {})["rag.generated_by"] == "claude"
    assert (finished["rag.retrieve"].attributes or {})["rag.n_results"] >= 1


def test_configure_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trace, "set_tracer_provider", lambda provider: None)
    assert tracing.configure_tracing("off") is False
    assert tracing.tracing_mode() == "off"
    assert tracing.configure_tracing("console") is True
    assert tracing.tracing_mode() == "console"
    assert tracing.configure_tracing("otlp") is True
    tracing.configure_tracing("off")


def test_attribute_cleaning() -> None:
    assert tracing._clean(["a", 1]) == ["a", "1"]
    assert tracing._clean({"k": 1}) == "{'k': 1}"
    assert tracing._clean(2.5) == 2.5
