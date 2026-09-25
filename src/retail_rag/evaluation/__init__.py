"""Evaluation harness: golden set, retrieval/generation metrics, reports, CI gates."""

from .dataset import GoldenExample, RelevantEvidence, load_golden_set
from .report import render_markdown, write_reports
from .runner import EvalReport, ExampleResult, check_thresholds, run_evaluation

__all__ = [
    "EvalReport",
    "ExampleResult",
    "GoldenExample",
    "RelevantEvidence",
    "check_thresholds",
    "load_golden_set",
    "render_markdown",
    "run_evaluation",
    "write_reports",
]
