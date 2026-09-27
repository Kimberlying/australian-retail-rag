"""OpenTelemetry tracing with a zero-cost no-op default.

``RAG_TRACING=otlp`` exports spans over OTLP/HTTP to whatever the standard
``OTEL_EXPORTER_OTLP_*`` variables point at: Jaeger, Grafana Tempo, Honeycomb,
or Langfuse (which ingests OTLP and renders the ``gen_ai.*`` attributes as LLM
generations with token usage). ``RAG_TRACING=console`` prints spans for local
debugging. With tracing off, or without the ``otel`` extra installed, ``span()``
yields a no-op and adds no measurable overhead.

Span layout for one request::

    rag.query                     route, retriever, refused, generated_by
    ├── rag.route                 route chosen by the router
    ├── rag.retrieve              top_k, filters, n_results, evidence_score
    ├── gen_ai.generate           gen_ai.* model + token usage (RAG answer)
    └── rag.sql_agent             tool-use loop over the read-only SQL tool
        └── rag.sql.execute       one span per statement Claude runs
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from .config import TracingMode

logger = logging.getLogger(__name__)

_TRACER_NAME = "retail_rag"
_configured_mode: TracingMode = "off"


class SpanLike(Protocol):
    def set_attribute(self, key: str, value: Any) -> None: ...


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None


_NOOP = _NoopSpan()


def configure_tracing(mode: TracingMode, *, service_name: str = "australian-retail-rag") -> bool:
    """Install a global tracer provider. Returns whether tracing is active."""
    global _configured_mode  # noqa: PLW0603 - process-wide telemetry switch
    if mode == "off":
        _configured_mode = "off"
        return False
    try:
        from opentelemetry import trace  # noqa: PLC0415 - optional dependency
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import (  # noqa: PLC0415
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SpanExporter,
        )
    except ImportError:
        logger.warning("RAG_TRACING=%s but the 'otel' extra is not installed", mode)
        _configured_mode = "off"
        return False

    exporter: SpanExporter
    if mode == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter()  # endpoint/headers from OTEL_EXPORTER_OTLP_* env vars
    else:
        exporter = ConsoleSpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _configured_mode = mode
    return True


def _clean(value: Any) -> Any:
    """OTel attributes accept str/bool/int/float and homogeneous lists of those."""
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple) and all(isinstance(v, str | int | float) for v in value):
        return [str(v) for v in value]
    return str(value)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[SpanLike]:
    """Open a span if tracing is active (or an SDK is installed by the host app)."""
    try:
        from opentelemetry import trace  # noqa: PLC0415
    except ImportError:
        yield _NOOP
        return
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, _clean(value))
        yield current


def set_attributes(target: SpanLike, **attributes: Any) -> None:
    for key, value in attributes.items():
        if value is not None:
            target.set_attribute(key, _clean(value))


def record_llm_usage(target: SpanLike, *, model: str, usage: dict[str, int]) -> None:
    """GenAI semantic-convention attributes, understood by Langfuse and most LLM UIs."""
    set_attributes(
        target,
        **{
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": usage.get("input_tokens"),
            "gen_ai.usage.output_tokens": usage.get("output_tokens"),
            "gen_ai.usage.cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "gen_ai.usage.cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        },
    )


def tracing_mode() -> TracingMode:
    return _configured_mode
