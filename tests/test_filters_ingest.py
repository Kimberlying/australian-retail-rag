from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from retail_rag.config import REPO_ROOT, Settings
from retail_rag.filters import companies, corpus_companies, describe, extract_filters, fiscal_years
from retail_rag.ingest import (
    load_chunks,
    parse_front_matter,
    read_document,
    strip_repeated_lines,
)
from retail_rag.pipeline import RAGPipeline
from retail_rag.retrieval import BM25Retriever

from .conftest import DOCS_DIR


def _report_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_sample_report", REPO_ROOT / "scripts" / "make_sample_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFiscalYears:
    @pytest.mark.parametrize(
        ("question", "years"),
        [
            ("What was Coles' FY25 EBIT?", [2025]),
            ("Revenue in FY2024 vs FY 25?", [2024, 2025]),
            ("sales for the 2024-25 financial year", [2025]),
            ("sales for 2023/24", [2024]),
            ("results for financial year 2025", [2025]),
            ("Coles revenue last financial year", []),  # relative: no filter, no guess
            ("How many stores opened in 2019?", []),  # bare calendar year: ambiguous
        ],
    )
    def test_extracts_only_explicit_fiscal_years(self, question: str, years: list[int]) -> None:
        assert fiscal_years(question) == years


class TestCompanies:
    def test_known_and_corpus_companies_with_possessives(self) -> None:
        assert companies("What was Coles' EBIT?") == ["Coles"]
        assert companies("Compare Woolworths and Coles") == ["Coles", "Woolworths"]
        assert companies("Harbourline's loyalty tiers") == ["Harbourline Retail"]
        assert companies("How does Acme Foods pay suppliers?", ["Acme Foods"]) == ["Acme Foods"]

    def test_word_boundaries(self) -> None:
        assert companies("the coleslaw recall") == []
        assert companies("Is IGA open?") == ["IGA"]
        assert companies("a big wholesale order") == []

    def test_extract_filters_and_describe(self) -> None:
        where = extract_filters("What was Coles' FY24 group sales revenue?")
        assert where == {"company": ["Coles"], "fiscal_year": [2024, None]}
        assert describe(where) == "Coles FY24"
        assert extract_filters("How quickly must recall stock be removed?") == {}

    def test_corpus_companies_reads_scalar_and_list_metadata(self) -> None:
        assert corpus_companies([{"company": ["A", "B"]}, {"company": "C"}, {}]) == {"A", "B", "C"}


class TestMetadataRefusal:
    @pytest.fixture
    def pipeline(self, settings: Settings) -> RAGPipeline:
        return RAGPipeline(BM25Retriever(load_chunks(DOCS_DIR)), settings)

    @pytest.mark.parametrize(
        "question",
        [
            "What was Woolworths' FY25 group EBIT?",  # company not in the corpus
            "What was Coles' FY24 group sales revenue?",  # period not in the corpus
            "What was Aldi's FY25 revenue?",  # another company that is not in the corpus
        ],
    )
    def test_refuses_without_llm_when_filters_match_nothing(
        self, pipeline: RAGPipeline, question: str
    ) -> None:
        result = pipeline.ask(question)
        assert result.refused
        assert result.refusal_reason == "metadata_filter"
        assert result.citations == []
        assert "No indexed document covers" in result.answer

    def test_filters_restrict_but_still_answer(self, pipeline: RAGPipeline) -> None:
        result = pipeline.ask("What was Harbourline's FY24 sales revenue?")
        assert not result.refused
        assert result.filters == {"company": ["Harbourline Retail"], "fiscal_year": [2024, None]}
        # FY24 is a comparative inside the FY25 annual report, which covers both years.
        top = result.citations[0]
        assert top["source"] == "harbourline_fy25_annual_report.pdf"
        assert top["page"] is not None
        assert "coles_fy25_public_snapshot.md" not in {c["source"] for c in result.citations}

    def test_fiscal_year_filter_keeps_undated_policies(self, pipeline: RAGPipeline) -> None:
        result = pipeline.ask(
            "How did Harbourline's FY25 out-of-stock rate compare with the replenishment "
            "policy target?"
        )
        sources = {citation["source"] for citation in result.citations}
        assert "harbourline_fy25_annual_report.pdf" in sources  # dated FY25
        assert "synthetic_inventory_replenishment_policy.md" in sources  # undated, kept

    def test_filters_can_be_disabled(self, settings: Settings) -> None:
        off = settings.model_copy(update={"metadata_filters": False})
        pipeline = RAGPipeline(BM25Retriever(load_chunks(DOCS_DIR)), off)
        result = pipeline.ask("What was Coles' FY24 group sales revenue?")
        assert result.filters == {}
        assert not result.refused  # BM25 has no gate: it happily retrieves the FY25 figures


class TestFrontMatterAndSidecar:
    def test_parse_front_matter(self) -> None:
        metadata, body = parse_front_matter(
            "---\ncompany: Coles\nfiscal_year: 2025\nyears: 2024, 2025\n---\n# Title\n"
        )
        assert metadata == {"company": "Coles", "fiscal_year": 2025, "years": [2024, 2025]}
        assert body == "# Title\n"
        assert parse_front_matter("# No front matter") == ({}, "# No front matter")

    def test_metadata_lists_and_sidecar_override(self, tmp_path: Path) -> None:
        doc = tmp_path / "note.md"
        doc.write_text("---\ncompany: Coles\ndoc_type: note\n---\nBody text here.", "utf-8")
        (tmp_path / "note.md.meta.json").write_text(json.dumps({"doc_type": "memo"}), "utf-8")
        pages, metadata = read_document(doc)
        assert pages[0].text == "Body text here."
        assert metadata == {"company": ["Coles"], "doc_type": "memo"}
        chunk = load_chunks(tmp_path)[0]
        assert chunk.metadata["company"] == ["Coles"]
        assert "page" not in chunk.metadata

    def test_unsupported_type(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            read_document(tmp_path / "data.csv")


class TestPdf:
    def test_strip_repeated_lines_removes_headers_and_footers(self) -> None:
        pages = [
            f"Acme Report 2025\nSection {n}\nNet debt {n}.0\nBody text {n}\nPage {n} of 3"
            for n in range(1, 4)
        ]
        # The running header and footer go; "Net debt" repeats too, but in the body.
        assert strip_repeated_lines(pages) == [
            f"Section {n}\nNet debt {n}.0\nBody text {n}" for n in range(1, 4)
        ]
        assert strip_repeated_lines(pages[:2]) == pages[:2]  # too few pages to judge

    def test_page_aware_chunks_from_generated_pdf(self, tmp_path: Path) -> None:
        report = _report_module()
        pdf = tmp_path / "report.pdf"
        report.write_pdf(
            pdf, [["Alpha revenue was A$5 million."], ["Beta stores: 12."], ["Gamma."]]
        )
        (tmp_path / "report.pdf.meta.json").write_text(
            json.dumps({"company": "Acme", "fiscal_year": 2025}), "utf-8"
        )
        chunks = load_chunks(tmp_path)
        assert [chunk.metadata["page"] for chunk in chunks] == [1, 2, 3]
        assert chunks[1].text == "Beta stores: 12."  # running header/footer stripped
        assert chunks[0].metadata["fiscal_year"] == [2025]

    def test_committed_report_matches_generator(self) -> None:
        report = _report_module()
        pages, metadata = read_document(DOCS_DIR / "harbourline_fy25_annual_report.pdf")
        assert metadata == {
            key: value if isinstance(value, list) else [value]
            for key, value in report.METADATA.items()
            if key in {"company", "fiscal_year"}
        } | {k: v for k, v in report.METADATA.items() if k not in {"company", "fiscal_year"}}
        assert len(pages) == len(report.PAGES)
        assert all(report.HEADER not in page.text for page in pages)
