"""Render evaluation reports as JSON (machine-readable) and Markdown (human/PR-readable)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import metrics

if TYPE_CHECKING:
    from .runner import EvalReport

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

    lines += _render_refusal_sweep(report)

    failures = [item for item in report.results if not item.passed]
    lines += ["", f"## Failures ({len(failures)})", ""]
    if not failures:
        lines.append("None.")
    for item in failures:
        top = item.retrieved[0]["source"] if item.retrieved else "nothing retrieved"
        reason = []
        if not item.answerable:
            reason.append("should have refused")
        else:
            if not item.hit:
                reason.append("relevant evidence not retrieved")
            if item.refused:
                reason.append("refused an answerable question")
            if item.answer_correct is False and not item.refused:
                reason.append("answer missing expected facts")
        lines.append(
            f"- **{item.id}** ({item.category}) — {'; '.join(reason)}. "
            f"Top hit: `{top}`. Q: _{item.question}_"
        )
    return "\n".join(lines) + "\n"


_SWEEP_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


def _render_refusal_sweep(report: EvalReport) -> list[str]:
    """Show how an evidence-gate threshold trades refusals against false refusals."""
    scored = [item for item in report.results if item.top_relevance is not None]
    if not scored:
        return []
    answerable = [item.top_relevance or 0.0 for item in report.results if item.answerable]
    unanswerable = [item.top_relevance or 0.0 for item in report.results if not item.answerable]
    if not unanswerable:
        return []
    gate = report.config.get("min_score")
    lines = [
        "",
        "## Evidence-gate threshold sweep",
        "",
        "Best-chunk relevance per question, measured before the gate. A question is",
        f"refused when it falls below the threshold (this run: `{gate}`).",
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
