"""Query orchestration: route -> filter -> retrieve -> gate -> generate (or query SQL).

Refusal is layered, cheapest check first, and every refusal records why:

1. ``metadata_filter``: the question names a company or fiscal year that no
   indexed document covers (Woolworths, FY24), so retrieval returns nothing.
2. ``evidence_gate``: the best chunk's dense similarity is below the calibrated
   threshold (clearly off-topic questions).
3. ``model``: Claude reads the evidence and finds it does not answer the question.

The first two cost no LLM call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .filters import corpus_companies, describe, extract_filters
from .generation import (
    REFUSAL_TEXT,
    Generation,
    build_context,
    generate_with_claude,
    local_preview,
    stream_with_claude,
)
from .models import Answer, GeneratedBy, RefusalReason, RetrievedChunk, Route
from .retrieval import Retriever, load_retriever
from .router import route_question
from .sql.agent import answer_with_sql
from .sql.tool import SQLTool
from .tracing import set_attributes, span

logger = logging.getLogger(__name__)

SQL_NEEDS_CLAUDE = (
    "This question needs the operational database. Answers from it are written by Claude, "
    "which is not configured (set ANTHROPIC_API_KEY); the read-only SQL tool itself is ready."
)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


@dataclass
class Retrieval:
    retrievals: list[RetrievedChunk]
    evidence_score: float | None
    where: dict[str, list[Any]] = field(default_factory=dict)
    refusal_reason: RefusalReason | None = None


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        settings: Settings | None = None,
        *,
        sql_tool: SQLTool | None = None,
    ):
        self.retriever = retriever
        self.settings = settings or get_settings()
        self.sql_tool = sql_tool
        self._companies = corpus_companies(chunk.metadata for chunk in retriever.chunks)

    @property
    def min_score(self) -> float:
        """Evidence-gate threshold: explicit setting, else the retriever's calibrated default."""
        if self.settings.min_score is not None:
            return self.settings.min_score
        return self.retriever.default_min_score

    # ------------------------------------------------------------------ steps

    def route(self, question: str) -> Route:
        if not self.settings.router_enabled:
            return "rag"
        with span("rag.route") as current:
            decision = route_question(question)
            route = decision.route
            if route != "rag" and self.sql_tool is None:
                route = "rag"  # no database built: documents are all we have
            set_attributes(current, **{"rag.route": route, "rag.route.reasons": decision.reasons})
        return route

    def retrieve(self, question: str, top_k: int) -> Retrieval:
        where = extract_filters(question, self._companies) if self.settings.metadata_filters else {}
        with span("rag.retrieve", **{"rag.top_k": top_k, "rag.filters": str(where)}) as current:
            retrievals = self.retriever.search(question, top_k=top_k, where=where or None)
            if where and not retrievals:
                set_attributes(current, **{"rag.refusal_reason": "metadata_filter"})
                return Retrieval([], None, where, "metadata_filter")
            # Answer-level evidence gate: refuse when even the best chunk is not similar
            # enough. Individual chunks are not filtered, so a hybrid result that BM25
            # found by exact term match survives even with modest dense similarity.
            relevances = [item.relevance for item in retrievals if item.relevance is not None]
            evidence_score = max(relevances) if relevances else None
            reason: RefusalReason | None = None
            if evidence_score is not None and evidence_score < self.min_score:
                retrievals, reason = [], "evidence_gate"
            elif not retrievals:
                reason = "evidence_gate"
            set_attributes(
                current,
                **{
                    "rag.n_results": len(retrievals),
                    "rag.evidence_score": evidence_score,
                    "rag.refusal_reason": reason,
                },
            )
        return Retrieval(retrievals, evidence_score, where, reason)

    # ------------------------------------------------------------------ answers

    def _refusal_text(self, found: Retrieval) -> str:
        if found.refusal_reason == "metadata_filter":
            return f"{REFUSAL_TEXT} No indexed document covers {describe(found.where)}."
        return REFUSAL_TEXT

    def ask(self, question: str, *, top_k: int | None = None) -> Answer:
        k = top_k or self.settings.top_k
        total_start = time.perf_counter()
        with span("rag.query", **{"rag.retriever": self.retriever.name}) as query_span:
            route = self.route(question)
            if route == "sql":
                answer = self._ask_sql(question, total_start)
            else:
                answer = self._ask_documents(question, k, route, total_start)
            set_attributes(
                query_span,
                **{
                    "rag.route": answer.route,
                    "rag.refused": answer.refused,
                    "rag.generated_by": answer.generated_by,
                },
            )
        self._log(answer)
        return answer

    def _ask_documents(self, question: str, k: int, route: Route, total_start: float) -> Answer:
        retrieval_start = time.perf_counter()
        found = self.retrieve(question, k)
        latency = {"retrieval": _elapsed_ms(retrieval_start)}

        generation_start = time.perf_counter()
        generated_by: GeneratedBy = "local"
        refused, reason = not found.retrievals, found.refusal_reason
        usage: dict[str, int] = {}
        queries: list[str] = []
        if not found.retrievals:
            answer_text = self._refusal_text(found)  # no LLM call spent
        elif route == "hybrid" and self.sql_tool is not None:
            answer_text, generated_by, refused, usage, queries = self._hybrid(question, found)
            reason = "model" if refused else None
        else:
            try:
                generation = generate_with_claude(question, found.retrievals, self.settings)
            except Exception:  # API/network errors must not take down retrieval.
                logger.exception("Claude generation failed; falling back to local preview")
                generation, generated_by = None, "local_fallback"
            if generation is not None:
                answer_text, refused, usage = generation.text, generation.refused, generation.usage
                generated_by = "claude"
                reason = "model" if refused else None
            else:
                answer_text = local_preview(question, found.retrievals)
        latency["generation"] = _elapsed_ms(generation_start)
        latency["total"] = _elapsed_ms(total_start)
        return Answer(
            question=question,
            answer=answer_text,
            citations=_citations(found.retrievals),
            retrievals=found.retrievals,
            generated_by=generated_by,
            refused=refused,
            evidence_score=found.evidence_score,
            latency_ms=latency,
            usage=usage,
            route=route,
            filters=found.where,
            refusal_reason=reason if refused else None,
            sql_queries=queries,
        )

    def _hybrid(
        self, question: str, found: Retrieval
    ) -> tuple[str, GeneratedBy, bool, dict[str, int], list[str]]:
        assert self.sql_tool is not None  # noqa: S101 - checked by the caller
        try:
            result = answer_with_sql(
                question, self.sql_tool, self.settings, documents=build_context(found.retrievals)
            )
        except Exception:
            logger.exception("SQL agent failed; falling back to the document preview")
            preview = local_preview(question, found.retrievals)
            return preview, "local_fallback", False, {}, []
        if result is None:
            preview = f"{local_preview(question, found.retrievals)}\n\n{SQL_NEEDS_CLAUDE}"
            return preview, "local", False, {}, []
        return result.text, "claude", result.refused, result.usage, result.queries

    def _ask_sql(self, question: str, total_start: float) -> Answer:
        assert self.sql_tool is not None  # noqa: S101 - route() only picks SQL when it exists
        generation_start = time.perf_counter()
        generated_by: GeneratedBy = "local"
        try:
            result = answer_with_sql(question, self.sql_tool, self.settings)
        except Exception:
            logger.exception("SQL agent failed")
            result, generated_by = None, "local_fallback"
        if result is None:
            text, refused, usage, queries = SQL_NEEDS_CLAUDE, False, {}, []
        else:
            text, refused, usage, queries = (
                result.text,
                result.refused,
                result.usage,
                result.queries,
            )
            generated_by = "claude"
        latency = {"retrieval": 0.0, "generation": _elapsed_ms(generation_start)}
        latency["total"] = _elapsed_ms(total_start)
        return Answer(
            question=question,
            answer=text,
            citations=[],
            retrievals=[],
            generated_by=generated_by,
            refused=refused,
            latency_ms=latency,
            usage=usage,
            route="sql",
            refusal_reason="model" if refused else None,
            sql_queries=queries,
        )

    # ------------------------------------------------------------------ streaming

    def stream(self, question: str, *, top_k: int | None = None) -> Iterator[dict[str, Any]]:
        """Yield ``meta``, then ``token`` deltas, then ``done`` with the full answer payload.

        Only document answers from Claude stream token by token; SQL and hybrid
        answers come from a tool-use loop, so they arrive as one token event.
        """
        k = top_k or self.settings.top_k
        total_start = time.perf_counter()
        route = self.route(question)
        if route != "rag":
            answer = self.ask(question, top_k=k)
            yield {"event": "meta", "route": answer.route, "citations": answer.citations}
            yield {"event": "token", "text": answer.answer}
            yield {"event": "done", **answer_payload(answer)}
            return

        retrieval_start = time.perf_counter()
        found = self.retrieve(question, k)
        latency = {"retrieval": _elapsed_ms(retrieval_start)}
        citations = _citations(found.retrievals)
        yield {"event": "meta", "route": route, "citations": citations}

        generation_start = time.perf_counter()
        generated_by: GeneratedBy = "local"
        generation: Generation | None = None
        text = ""
        if found.retrievals:
            try:
                for event in stream_with_claude(question, found.retrievals, self.settings):
                    if event.kind == "text":
                        text += event.text
                        yield {"event": "token", "text": event.text}
                    else:
                        generation = event.generation
            except Exception:
                logger.exception("Claude streaming failed; falling back to local preview")
                generated_by = "local_fallback"
        if generation is not None:
            generated_by = "claude"
            answer_text, refused, usage = generation.text, generation.refused, generation.usage
        else:
            answer_text = (
                local_preview(question, found.retrievals)
                if found.retrievals
                else self._refusal_text(found)
            )
            refused, usage = not found.retrievals, {}
            if not text:
                yield {"event": "token", "text": answer_text}
        latency["generation"] = _elapsed_ms(generation_start)
        latency["total"] = _elapsed_ms(total_start)
        answer = Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrievals=found.retrievals,
            generated_by=generated_by,
            refused=refused,
            evidence_score=found.evidence_score,
            latency_ms=latency,
            usage=usage,
            route=route,
            filters=found.where,
            refusal_reason=(found.refusal_reason or "model") if refused else None,
        )
        self._log(answer)
        yield {"event": "done", **answer_payload(answer)}

    # ------------------------------------------------------------------ misc

    def _log(self, answer: Answer) -> None:
        logger.info(
            "query answered",
            extra={
                "fields": {
                    "route": answer.route,
                    "generated_by": answer.generated_by,
                    "refused": answer.refused,
                    "refusal_reason": answer.refusal_reason,
                    "filters": answer.filters,
                    "n_retrieved": len(answer.retrievals),
                    "retriever": self.retriever.name,
                    "evidence_score": answer.evidence_score,
                    "latency_ms": answer.latency_ms.get("total"),
                    **answer.usage,
                }
            },
        )

    @classmethod
    def from_index(cls, index_path: Path, settings: Settings | None = None) -> RAGPipeline:
        settings = settings or get_settings()
        sql_tool = SQLTool(settings.sql_db_path) if settings.sql_db_path.is_file() else None
        return cls(load_retriever(index_path, settings), settings, sql_tool=sql_tool)


def _citations(retrievals: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "source": item.chunk.source,
            "chunk_id": item.chunk.chunk_id,
            "page": item.chunk.metadata.get("page"),
            "score": round(item.score, 4),
            "relevance": None if item.relevance is None else round(item.relevance, 4),
        }
        for item in retrievals
    ]


def answer_payload(answer: Answer) -> dict[str, Any]:
    """The JSON shape shared by the CLI, ``/query``, and the final stream event."""
    return {
        "question": answer.question,
        "answer": answer.answer,
        "citations": answer.citations,
        "route": answer.route,
        "sql_queries": answer.sql_queries,
        "generated_by": answer.generated_by,
        "refused": answer.refused,
        "refusal_reason": answer.refusal_reason,
        "filters": answer.filters,
        "evidence_score": answer.evidence_score,
        "latency_ms": answer.latency_ms,
        "usage": answer.usage,
    }
