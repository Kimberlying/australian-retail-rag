from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from retail_rag.api import app
from retail_rag.config import get_settings
from retail_rag.ingest import build_index


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, docs_dir: Path, tmp_path: Path) -> Iterator[TestClient]:
    index = tmp_path / "index.json"
    build_index(docs_dir, index)
    monkeypatch.setenv("RAG_INDEX_PATH", str(index))
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setenv("RAG_INDEX_PATH", str(tmp_path / "missing.json"))
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_loaded_index(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["chunks"] > 0
    assert body["llm_enabled"] is False


def test_query_returns_answer_and_citations(client: TestClient) -> None:
    response = client.post(
        "/query", json={"question": "What was Coles normalised eCommerce sales growth in FY25?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "23.3" in body["answer"]
    assert body["citations"][0]["source"] == "coles_fy25_public_snapshot.md"
    assert body["generated_by"] == "local"
    assert body["latency_ms"]["total"] >= 0


def test_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health", headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"
    assert client.get("/health").headers["x-request-id"]


@pytest.mark.parametrize(
    "payload",
    [{"question": "hi"}, {"question": "x" * 1001}, {"question": "valid question", "top_k": 50}],
)
def test_rejects_invalid_requests(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/query", json=payload).status_code == 422


def test_not_ready_without_index(client_without_index: TestClient) -> None:
    assert client_without_index.get("/health").status_code == 200
    assert client_without_index.get("/ready").status_code == 503
    assert client_without_index.post("/query", json={"question": "anything?"}).status_code == 503
