from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from retail_rag.cli import main
from retail_rag.config import Settings
from retail_rag.evaluation import (
    GoldenExample,
    check_thresholds,
    load_golden_set,
    metrics,
    render_markdown,
    run_evaluation,
    write_reports,
)
from retail_rag.ingest import load_chunks
from retail_rag.models import DocumentChunk
from retail_rag.pipeline import RAGPipeline
from retail_rag.retrieval import TfidfRetriever

from .conftest import GOLDEN_SET


class TestMetrics:
    def test_hit_rate(self) -> None:
        assert metrics.hit_rate([False, True]) == 1.0
        assert metrics.hit_rate([False, False]) == 0.0
        assert metrics.hit_rate([]) == 0.0

    def test_recall_and_precision(self) -> None:
        assert metrics.recall([True, False, True, True]) == 0.75
        assert metrics.precision([True, False, False, False]) == 0.25
        assert metrics.recall([]) == 0.0

    def test_reciprocal_rank(self) -> None:
        assert metrics.reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
        assert metrics.reciprocal_rank([False]) == 0.0

    def test_ndcg_matches_hand_computation(self) -> None:
        # relevant at ranks 2 and 3, two relevant items overall
        expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 / math.log2(2) + 1 / math.log2(3))
        assert metrics.ndcg([False, True, True], n_relevant=2, k=3) == pytest.approx(expected)
        assert metrics.ndcg([True, True], n_relevant=2, k=2) == pytest.approx(1.0)
        assert metrics.ndcg([False], n_relevant=0, k=1) == 0.0

    def test_contains_all_is_case_insensitive(self) -> None:
        assert metrics.contains_all("Held in JANUARY and July", ["january", "July"])
        assert not metrics.contains_all("January only", ["January", "July"])

    def test_citation_validity_detects_hallucinated_sources(self) -> None:
        answer = "Sources: coles_fy25_public_snapshot.md, made_up_report.pdf"
        assert metrics.citation_validity(answer, ["coles_fy25_public_snapshot.md"]) == 0.5
        assert metrics.citation_validity("Sources: a.md", ["docs/a.md"]) == 1.0
        assert metrics.citation_validity("no citations here", ["a.md"]) is None

    def test_percentile_nearest_rank(self) -> None:
        values = [float(v) for v in range(1, 101)]
        assert metrics.percentile(values, 50) == 50
        assert metrics.percentile(values, 95) == 95
        assert metrics.percentile([], 95) == 0.0


class TestGoldenSet:
    def test_repository_golden_set_is_valid(self, docs_dir: Path) -> None:
        examples = load_golden_set(GOLDEN_SET, docs_dir=docs_dir)
        assert len(examples) >= 50
        categories = {example.category for example in examples}
        assert {"public_fact", "synthetic_policy", "paraphrase", "unanswerable"} <= categories

    def test_every_label_is_satisfiable_by_some_chunk(self, docs_dir: Path) -> None:
        """Guards against labels drifting from the corpus (typos, edited documents)."""
        chunks = load_chunks(docs_dir)
        for example in load_golden_set(GOLDEN_SET, docs_dir=docs_dir):
            for evidence in example.relevant:
                assert any(evidence.matches(c.source, c.text) for c in chunks), (
                    f"{example.id}: no chunk matches {evidence}"
                )

    def test_rejects_answerable_example_without_labels(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"id": "x", "question": "why?", "category": "public_fact"}))
        with pytest.raises(ValueError, match="at least one relevant"):
            load_golden_set(path)

    def test_rejects_duplicate_ids_and_unknown_sources(
        self, tmp_path: Path, docs_dir: Path
    ) -> None:
        row = {
            "id": "dup",
            "question": "why?",
            "category": "public_fact",
            "relevant": [{"source": "nope.md"}],
        }
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="duplicate ids") as excinfo:
            load_golden_set(path, docs_dir=docs_dir)
        assert "unknown source" in str(excinfo.value)


def _toy_examples() -> list[GoldenExample]:
    return [
        GoldenExample.model_validate(
            {
                "id": "a",
                "question": "ecommerce sales growth",
                "category": "public_fact",
                "relevant": [{"source": "sales.md", "contains": "23.3"}],
                "answer_must_contain": ["23.3"],
            }
        ),
        GoldenExample.model_validate(
            {"id": "b", "question": "zzz qqq", "category": "unanswerable", "answerable": False}
        ),
    ]


class TestRunner:
    def test_run_evaluation_on_toy_corpus(
        self, settings: Settings, toy_chunks: list[DocumentChunk]
    ) -> None:
        report = run_evaluation(
            RAGPipeline(TfidfRetriever(toy_chunks), settings), _toy_examples(), k=2
        )
        assert report.summary["hit_rate"] == 1.0
        assert report.summary["mrr"] == 1.0
        assert report.summary["answer_accuracy"] == 1.0
        assert report.summary["refusal_accuracy"] == 1.0
        assert report.summary["pass_rate"] == 1.0
        assert set(report.by_category) == {"public_fact", "unanswerable"}

    def test_reports_are_written_and_rendered(
        self, settings: Settings, toy_chunks: list[DocumentChunk], tmp_path: Path
    ) -> None:
        report = run_evaluation(
            RAGPipeline(TfidfRetriever(toy_chunks), settings), _toy_examples(), k=2
        )
        json_path, md_path = write_reports(report, tmp_path)
        assert json.loads(json_path.read_text())["summary"]["n_examples"] == 2
        markdown = render_markdown(report)
        assert "| Hit rate@k | 1.000 |" in markdown
        assert md_path.read_text() == markdown

    def test_check_thresholds(self) -> None:
        summary = {"recall": 0.8, "mrr": 0.9}
        assert check_thresholds(summary, {"recall": 0.8}) == []
        failures = check_thresholds(summary, {"mrr": 0.95, "faithfulness": 0.9})
        assert len(failures) == 2
        assert any("not measured" in failure for failure in failures)


class TestEvalCli:
    def test_passes_gate(self, tmp_path: Path) -> None:
        code = main(["eval", "--output-dir", str(tmp_path), "--fail-under", "hit_rate=0.5"])
        assert code == 0
        assert (tmp_path / "eval.json").exists()
        assert (tmp_path / "eval.md").exists()

    def test_fails_gate(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["eval", "--output-dir", str(tmp_path), "--fail-under", "hit_rate=1.01"])
        assert code == 1
        assert "Quality gate FAILED" in capsys.readouterr().err


class TestClaudeModeEvaluation:
    """Exercise citation validity and the LLM judge without network calls."""

    def test_judge_and_citation_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        settings: Settings,
        toy_chunks: list[DocumentChunk],
    ) -> None:
        from types import SimpleNamespace
        from typing import Any

        import anthropic
        from pydantic import SecretStr

        from retail_rag.evaluation.judge import FaithfulnessJudge, FaithfulnessVerdict

        verdict = FaithfulnessVerdict(
            verdict="partially_supported", reasoning="One claim unsupported."
        )

        class FakeClient:
            def __init__(self, **_: Any) -> None:
                reply = SimpleNamespace(
                    model="m",
                    stop_reason="end_turn",
                    content=[
                        SimpleNamespace(type="text", text="23.3%. Sources: sales.md, fake.pdf")
                    ],
                    usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                )
                self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: reply))
                self.messages = SimpleNamespace(
                    parse=lambda **_: SimpleNamespace(parsed_output=verdict, stop_reason="end_turn")
                )

        monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
        llm_settings = settings.model_copy(update={"anthropic_api_key": SecretStr("k")})
        pipeline = RAGPipeline(TfidfRetriever(toy_chunks), llm_settings)

        report = run_evaluation(
            pipeline, _toy_examples()[:1], k=2, judge=FaithfulnessJudge(llm_settings)
        )
        result = report.results[0]
        assert result.generated_by == "claude"
        assert result.citation_validity == 0.5  # fake.pdf was never retrieved
        assert result.faithfulness == 0.5
        assert report.summary["total_output_tokens"] == 5

    def test_judge_requires_api_key(self, settings: Settings) -> None:
        from retail_rag.evaluation.judge import FaithfulnessJudge

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            FaithfulnessJudge(settings)
