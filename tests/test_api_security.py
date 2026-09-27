from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from retail_rag.api import app
from retail_rag.config import get_settings
from retail_rag.ingest import build_index
from retail_rag.security import APIKeyAuth, RateLimiter, key_fingerprint
from retail_rag.sql import build_database

QUESTION = {"question": "How quickly must Class A recall stock be removed?"}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestAPIKeyAuth:
    def test_disabled_without_keys(self) -> None:
        auth = APIKeyAuth([])
        assert not auth.enabled
        assert auth.verify(None)

    def test_verifies_any_configured_key(self) -> None:
        auth = APIKeyAuth(["alpha", "beta"])
        assert auth.verify("beta")
        assert not auth.verify("gamma")
        assert not auth.verify("")
        assert not auth.verify(None)

    def test_fingerprint_does_not_reveal_key(self) -> None:
        assert "secret" not in key_fingerprint("secret")
        assert len(key_fingerprint("secret")) == 12


class TestRateLimiter:
    def test_burst_then_refill(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(per_minute=60, burst=2, monotonic=clock)
        assert limiter.acquire("a") == 0
        assert limiter.acquire("a") == 0
        wait = limiter.acquire("a")
        assert wait == pytest.approx(1.0)  # one token per second
        assert RateLimiter.retry_after_header(wait) == "1"
        assert limiter.acquire("b") == 0  # buckets are per client
        clock.now += 1.0
        assert limiter.acquire("a") == 0

    def test_disabled(self) -> None:
        limiter = RateLimiter(per_minute=0, burst=1)
        assert not limiter.enabled
        assert all(limiter.acquire("a") == 0 for _ in range(100))

    def test_prunes_idle_clients(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(per_minute=60, burst=1, monotonic=clock)
        limiter.max_clients = 2
        limiter.acquire("a")
        limiter.acquire("b")
        clock.now += 10  # both buckets have refilled
        limiter.acquire("c")
        assert set(limiter._buckets) == {"c"}


@pytest.fixture
def make_client(
    monkeypatch: pytest.MonkeyPatch, docs_dir: Path, tmp_path: Path
) -> Iterator[Callable[..., TestClient]]:
    index = tmp_path / "index.json"
    build_index(docs_dir, index)
    db = build_database(tmp_path / "retail.db")
    monkeypatch.setenv("RAG_INDEX_PATH", str(index))
    monkeypatch.setenv("RAG_SQL_DB_PATH", str(db))
    clients: list[TestClient] = []

    def factory(**env: str) -> TestClient:
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        get_settings.cache_clear()
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


class TestProtectedEndpoints:
    def test_auth_required_when_keys_configured(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client(RAG_API_KEYS="k1, k2")
        assert client.post("/query", json=QUESTION).status_code == 401
        wrong = client.post("/query", json=QUESTION, headers={"x-api-key": "nope"})
        assert wrong.status_code == 401
        assert wrong.headers["www-authenticate"] == "Bearer"
        assert client.post("/query", json=QUESTION, headers={"x-api-key": "k2"}).status_code == 200
        bearer = client.post("/query", json=QUESTION, headers={"Authorization": "Bearer k1"})
        assert bearer.status_code == 200
        assert client.get("/health").status_code == 200  # probes stay open
        assert client.get("/ready").json()["auth_enabled"] is True

    def test_rate_limit_returns_429_with_retry_after(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client(RAG_RATE_LIMIT_PER_MINUTE="1", RAG_RATE_LIMIT_BURST="2")
        assert client.post("/query", json=QUESTION).status_code == 200
        assert client.post("/query", json=QUESTION).status_code == 200
        limited = client.post("/query", json=QUESTION)
        assert limited.status_code == 429
        assert int(limited.headers["retry-after"]) >= 1

    def test_failed_auth_attempts_are_rate_limited(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client(
            RAG_API_KEYS="k1", RAG_RATE_LIMIT_PER_MINUTE="1", RAG_RATE_LIMIT_BURST="2"
        )
        codes = [
            client.post("/query", json=QUESTION, headers={"x-api-key": f"guess{n}"}).status_code
            for n in range(3)
        ]
        assert codes == [401, 401, 429]

    def test_query_reports_route_filters_and_pages(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client()
        body = client.post(
            "/query", json={"question": "What was Harbourline's FY25 sales revenue?"}
        ).json()
        assert body["route"] == "rag"
        assert body["filters"] == {"company": ["Harbourline Retail"], "fiscal_year": [2025, None]}
        assert body["citations"][0]["page"] is not None
        refused = client.post("/query", json={"question": "What was Coles' FY24 revenue?"}).json()
        assert refused["refused"] is True
        assert refused["refusal_reason"] == "metadata_filter"
        sql = client.post("/query", json={"question": "How many orders did HB001 have in July?"})
        assert sql.json()["route"] == "sql"
        assert client.get("/ready").json()["sql_enabled"] is True

    def test_stream_emits_meta_tokens_and_done(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client()
        with client.stream("POST", "/query/stream", json=QUESTION) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            raw = "".join(response.iter_text())
        events = [
            (block.split("\n")[0].removeprefix("event: "), json.loads(block.split("data: ", 1)[1]))
            for block in raw.strip().split("\n\n")
        ]
        names = [name for name, _ in events]
        assert names[0] == "meta"
        assert names[-1] == "done"
        assert "token" in names
        assert events[0][1]["citations"]
        assert events[-1][1]["generated_by"] == "local"
        assert "2 hours" in events[-1][1]["answer"]
