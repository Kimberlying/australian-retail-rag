from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from retail_rag.cli import main
from retail_rag.logging_config import JsonFormatter


def test_ingest_then_query_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index = tmp_path / "index.json"
    assert main(["ingest", "--index", str(index)]) == 0
    capsys.readouterr()

    assert main(["query", "What was Coles FY25 EBIT?", "--index", str(index), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["citations"][0]["source"] == "coles_fy25_public_snapshot.md"
    assert payload["generated_by"] == "local"


def test_query_without_index_exits_2(tmp_path: Path) -> None:
    assert main(["query", "anything", "--index", str(tmp_path / "missing.json")]) == 2


def test_rejects_malformed_threshold() -> None:
    with pytest.raises(SystemExit):
        main(["eval", "--fail-under", "recall"])


def test_json_log_formatter_includes_structured_fields() -> None:
    record = logging.LogRecord("retail_rag.x", logging.INFO, __file__, 1, "hello", None, None)
    record.fields = {"request_id": "r1", "latency_ms": 3.2}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello"
    assert payload["request_id"] == "r1"
    assert payload["level"] == "INFO"
