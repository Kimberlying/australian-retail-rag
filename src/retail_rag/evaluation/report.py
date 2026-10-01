"""Render evaluation reports as JSON (machine-readable) and Markdown (human/PR-readable)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import metrics

if TYPE_CHECKING:
    from .runner import EvalReport, ExampleResult

_HEADLINE = [
    ("hit_rate", "Hit rate@k"),
    ("recall", "Recall@k"),
    ("mrr", "MRR"),
    ("ndcg", "nDCG@k"),
    ("precision", "Precision@k"),
    ("answer_accuracy", "Answer accuracy"),
    ("refusal_accuracy", "Refusal accuracy (unanswerable)"),
    ("false_refusal_rate", "False refusal rate (answerable)"),
    ("citation_validity", "Citation validity"),
    ("faithfulness", "Faithfulness (LLM judge)"),
    ("route_accuracy", "Route accuracy (RAG / SQL / hybrid)"),
    ("route_accuracy_holdout", "Route accuracy, held-out set"),
    ("sql_execution_accuracy", "SQL execution accuracy"),
    ("pass_rate", "Overall pass rate"),
    ("latency_p50_ms", "Latency p50 (ms)"),
    ("latency_p95_ms", "Latency p95 (ms)"),
]
_CATEGORY_COLUMNS = ["n_examples", "hit_rate", "recall", "mrr", "refusal_accuracy", "pass_rate"]


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if float(value).is_integer() and value > 1:
        return str(int(value))
    return f"{value:.3f}"


def render_markdown(report: EvalReport) -> str:
    modes = sorted({item.generated_by for item in report.results})
    lines = [
        "# RAG evaluation report",
        "",
        f"- **Created:** {report.created_at}",
        f"- **Git SHA:** `{report.git_sha or 'unknown'}` · **version:** {report.version}",
        f"- **Generation mode:** {', '.join(modes)}",
        "- **Config:** " + ", ".join(f"`{key}={value}`" for key, value in report.config.items()),
        "",
    ]
    if modes == ["local"]:
        lines += [
            "> Generation ran in local-preview mode (no `ANTHROPIC_API_KEY`), so answer",
            "> metrics reflect retrieved evidence only and refusals come solely from the",
            "> retrieval score threshold.",
            "",
        ]

    lines += ["## Summary", "", "| Metric | Value |", "|---|---|"]
    for key, label in _HEADLINE:
        if key in report.summary:
            lines.append(f"| {label} | {_fmt(report.summary[key])} |")

    lines += [
        "",
        "## By category",
        "",
        "| Category | " + " | ".join(_CATEGORY_COLUMNS) + " |",
        "|---|" + "---|" * len(_CATEGORY_COLUMNS),
    ]
    for name, stats in report.by_category.items():
        cells = [_fmt(stats.get(column)) for column in _CATEGORY_COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += _render_refusal_layers(report)
    lines += _render_routing(report)
    lines += _render_refusal_sweep(report)

    failures = [item for item in report.results if not item.passed]
    lines += ["", f"## Failures ({len(failures)})", ""]
    if not failures:
        lines.append("None.")
    for item in failures:
        top = item.retrieved[0]["source"] if item.retrieved else "nothing retrieved"
        lines.append(
            f"- **{item.id}** ({item.category}) — {'; '.join(_failure_reasons(item))}. "
            f"Top hit: `{top}`. Q: _{item.question}_"
        )
    return "\n".join(lines) + "\n"


def _failure_reasons(item: ExampleResult) -> list[str]:
    reasons = []
    if item.route_correct is False:
        reasons.append(f"routed to {item.route}, expected {item.expected_route}")
    if item.sql_correct is False:
        reasons.append("SQL result does not match the gold query")
    if item.expected_route == "sql":
        return reasons
    if not item.answerable:
        return [*reasons, "should have refused"]
    if not item.hit:
        reasons.append("relevant evidence not retrieved")
    if item.refused:
        reasons.append("refused an answerable question")
    if item.answer_correct is False and not item.refused:
        reasons.append("answer missing expected facts")
    return reasons


_SWEEP_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


def _render_refusal_layers(report: EvalReport) -> list[str]:
    """Which layer refused each unanswerable document question."""
    pool = [item for item in report.results if not item.answerable and item.expected_route == "rag"]
    if not pool:
        return []
    counts: dict[str, int] = {}
    for item in pool:
        key = item.refusal_reason if item.refused else "answered (missed)"
        counts[key or "unknown"] = counts.get(key or "unknown", 0) + 1
    lines = [
        "",
        "## Refusal layers (unanswerable questions)",
        "",
        "| Layer | Questions |",
        "|---|---|",
    ]
    for key in ("metadata_filter", "evidence_gate", "model", "answered (missed)"):
        if key in counts:
            lines.append(f"| {key} | {counts[key]} |")
    return lines


def _render_routing(report: EvalReport) -> list[str]:
    graded = [item for item in report.results if item.route_correct is not None]
    if not graded:
        return []
    routes = ["rag", "sql", "hybrid"]
    lines = [
        "",
        "## Routing",
        "",
        "Rows are the labelled route, columns the route the router chose.",
        "",
        "| Expected \\ Chosen | " + " | ".join(routes) + " |",
        "|---|" + "---|" * len(routes),
    ]
    for expected in routes:
        row = [
            sum(1 for item in graded if item.expected_route == expected and item.route == chosen)
            for chosen in routes
        ]
        if any(row):
            lines.append(f"| {expected} | " + " | ".join(str(count) for count in row) + " |")
    if "route_accuracy_holdout" in report.summary:
        lines += [
            "",
            "The router's rules were written against the golden set, so the table above is a",
            "training-set number. On the held-out set (`evals/router_holdout.jsonl`, never used",
            f"for tuning) accuracy is **{report.summary['route_accuracy_holdout']:.3f}**.",
        ]
        lines += [
            f"- held-out miss: expected {miss['expected']}, chose {miss['chosen']}: "
            f"_{miss['question']}_"
            for miss in report.router_holdout_misses
        ]
    return lines


def _render_refusal_sweep(report: EvalReport) -> list[str]:
    """Show how an evidence-gate threshold trades refusals against false refusals."""
    # Only questions that reached the gate: metadata-filter refusals never did.
    gated = [
        item
        for item in report.results
        if item.expected_route == "rag"
        and item.route == "rag"
        and item.refusal_reason != "metadata_filter"
    ]
    if not any(item.top_relevance is not None for item in gated):
        return []
    answerable = [item.top_relevance or 0.0 for item in gated if item.answerable]
    unanswerable = [item.top_relevance or 0.0 for item in gated if not item.answerable]
    if not unanswerable:
        return []
    gate = report.config.get("min_score")
    lines = [
        "",
        "## Evidence-gate threshold sweep",
        "",
        "Best-chunk relevance per question, measured before the gate. A question is",
        f"refused when it falls below the threshold (this run: `{gate}`). Questions",
        "already refused by the metadata filter never reach the gate and are excluded.",
        "",
        f"- Answerable: min {min(answerable):.3f}, median {metrics.percentile(answerable, 50):.3f}",
        f"- Unanswerable: max {max(unanswerable):.3f}, "
        f"median {metrics.percentile(unanswerable, 50):.3f}",
        "",
        "| Threshold | Refusal accuracy | False refusal rate |",
        "|---|---|---|",
    ]
    for row in metrics.refusal_sweep(answerable, unanswerable, _SWEEP_THRESHOLDS):
        lines.append(
            f"| {row['threshold']:.2f} | {row['refusal_accuracy']:.3f} "
            f"| {row['false_refusal_rate']:.3f} |"
        )
    return lines


def write_reports(report: EvalReport, output_dir: Path, *, stem: str = "eval") -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
