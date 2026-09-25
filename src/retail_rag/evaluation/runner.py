"""Run the golden set through the pipeline and aggregate metrics."""

from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .. import __version__
from ..pipeline import RAGPipeline
from . import metrics

if TYPE_CHECKING:
    from .dataset import GoldenExample
    from .judge import FaithfulnessJudge

logger = logging.getLogger(__name__)


class ExampleResult(BaseModel):
    id: str
    category: str
    question: str
    answerable: bool
    retrieved: list[dict[str, Any]]
    # retrieval (answerable only)
    hit: float | None = None
    recall: float | None = None
    precision: float | None = None
    reciprocal_rank: float | None = None
    ndcg: float | None = None
    # generation
    answer: str
    generated_by: str
    refused: bool
    answer_correct: bool | None = None
    citation_validity: float | None = None
    faithfulness: float | None = None
    judge_notes: str | None = None
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def passed(self) -> bool:
        if not self.answerable:
            return self.refused
        return bool(self.hit) and not self.refused and self.answer_correct is not False


class EvalReport(BaseModel):
    created_at: str
    git_sha: str | None
    version: str
    config: dict[str, Any]
    summary: dict[str, float] = Field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    results: list[ExampleResult] = Field(default_factory=list)


def _git_sha() -> str | None:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607 - fixed argv, no user input
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return output.stdout.strip() or None


def evaluate_example(
    pipeline: RAGPipeline,
    example: GoldenExample,
    *,
    k: int,
    judge: FaithfulnessJudge | None = None,
) -> ExampleResult:
    answer = pipeline.ask(example.question, top_k=k)
    chunks = [(item.chunk.source, item.chunk.text) for item in answer.retrievals]
    relevance = [
        any(evidence.matches(source, text) for evidence in example.relevant)
        for source, text in chunks
    ]
    found = [
        any(evidence.matches(source, text) for source, text in chunks)
        for evidence in example.relevant
    ]

    result = ExampleResult(
        id=example.id,
        category=example.category,
        question=example.question,
        answerable=example.answerable,
        retrieved=answer.citations,
        answer=answer.answer,
        generated_by=answer.generated_by,
        refused=answer.refused,
        latency_ms=answer.latency_ms.get("total", 0.0),
        input_tokens=answer.usage.get("input_tokens", 0),
        output_tokens=answer.usage.get("output_tokens", 0),
    )
    if example.answerable:
        result.hit = metrics.hit_rate(relevance)
        result.recall = metrics.recall(found)
        result.precision = metrics.precision(relevance)
        result.reciprocal_rank = metrics.reciprocal_rank(relevance)
        result.ndcg = metrics.ndcg(relevance, n_relevant=len(example.relevant), k=k)
        if example.answer_must_contain:
            result.answer_correct = not answer.refused and metrics.contains_all(
                answer.answer, example.answer_must_contain
            )

    if answer.generated_by == "claude":
        result.citation_validity = metrics.citation_validity(
            answer.answer, [source for source, _ in chunks]
        )
        if judge is not None and answer.retrievals:
            verdict = judge.grade(example.question, answer.answer, answer.retrievals)
            result.faithfulness = verdict.score
            result.judge_notes = verdict.reasoning
    return result


def _aggregate(results: list[ExampleResult]) -> dict[str, float]:
    answerable = [item for item in results if item.answerable]
    unanswerable = [item for item in results if not item.answerable]

    def avg(field: str, pool: list[ExampleResult]) -> float | None:
        values = [getattr(item, field) for item in pool if getattr(item, field) is not None]
        return round(metrics.mean([float(value) for value in values]), 4) if values else None

    summary: dict[str, float | None] = {
        "n_examples": len(results),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "hit_rate": avg("hit", answerable),
        "recall": avg("recall", answerable),
        "precision": avg("precision", answerable),
        "mrr": avg("reciprocal_rank", answerable),
        "ndcg": avg("ndcg", answerable),
        "answer_accuracy": avg("answer_correct", answerable),
        "false_refusal_rate": avg("refused", answerable) if answerable else None,
        "refusal_accuracy": avg("refused", unanswerable) if unanswerable else None,
        "citation_validity": avg("citation_validity", results),
        "faithfulness": avg("faithfulness", results),
        "pass_rate": round(metrics.mean([float(item.passed) for item in results]), 4),
        "latency_p50_ms": metrics.percentile([item.latency_ms for item in results], 50),
        "latency_p95_ms": metrics.percentile([item.latency_ms for item in results], 95),
        "total_input_tokens": sum(item.input_tokens for item in results),
        "total_output_tokens": sum(item.output_tokens for item in results),
    }
    return {key: value for key, value in summary.items() if value is not None}


def run_evaluation(
    pipeline: RAGPipeline,
    examples: list[GoldenExample],
    *,
    k: int,
    judge: FaithfulnessJudge | None = None,
    config: dict[str, Any] | None = None,
) -> EvalReport:
    results = []
    for example in examples:
        results.append(evaluate_example(pipeline, example, k=k, judge=judge))
        logger.debug("evaluated", extra={"fields": {"id": example.id}})

    grouped: dict[str, list[ExampleResult]] = defaultdict(list)
    for item in results:
        grouped[item.category].append(item)

    return EvalReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        version=__version__,
        config={"k": k, **(config or {})},
        summary=_aggregate(results),
        by_category={name: _aggregate(items) for name, items in sorted(grouped.items())},
        results=results,
    )


def check_thresholds(summary: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    """Return human-readable failures for every metric below its minimum."""
    failures = []
    for name, minimum in thresholds.items():
        if name not in summary:
            failures.append(f"{name}: not measured in this run (required >= {minimum})")
        elif summary[name] < minimum:
            failures.append(f"{name}: {summary[name]:.4f} < required {minimum:.4f}")
    return failures
