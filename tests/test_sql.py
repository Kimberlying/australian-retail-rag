from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from retail_rag.config import Settings
from retail_rag.ingest import load_chunks
from retail_rag.pipeline import SQL_NEEDS_CLAUDE, RAGPipeline
from retail_rag.retrieval import BM25Retriever
from retail_rag.router import route_question
from retail_rag.sql import SQLTool, UnsafeQueryError, build_database
from retail_rag.sql.agent import DATA_REFUSAL_TEXT, answer_with_sql

from .conftest import DOCS_DIR
from .fakes import ScriptedAnthropic, install, message, text_block, tool_use_block


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_database(tmp_path_factory.mktemp("sql") / "retail.db")


@pytest.fixture
def tool(db_path: Path) -> SQLTool:
    return SQLTool(db_path)


@pytest.fixture
def llm_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"anthropic_api_key": SecretStr("test-key")})


class TestSyntheticDatabase:
    def test_is_deterministic(self, db_path: Path, tmp_path: Path) -> None:
        again = SQLTool(build_database(tmp_path / "again.db"))
        query = "SELECT COUNT(*), ROUND(SUM(line_total_aud), 2) FROM order_items"
        assert again.run(query).rows == SQLTool(db_path).run(query).rows

    def test_seeds_the_policy_scenarios(self, tool: SQLTool) -> None:
        at_risk = tool.run(
            "SELECT COUNT(*) FROM inventory_snapshots a JOIN inventory_snapshots b "
            "ON a.store_id = b.store_id AND a.sku = b.sku "
            "WHERE a.snapshot_date = '2026-09-01' AND b.snapshot_date = '2026-09-02' "
            "AND a.on_hand < a.reorder_point AND b.on_hand < b.reorder_point"
        ).rows[0][0]
        assert at_risk > 0
        shrink = tool.run(
            "SELECT store_id FROM shrink_weekly WHERE shrink_aud / sales_aud > 0.018 "
            "GROUP BY store_id HAVING COUNT(*) >= 3"
        ).rows
        assert shrink == [("HB006",)]
        assert tool.run("SELECT COUNT(*) FROM stores WHERE state = 'WA'").rows == [(0,)]


class TestSQLTool:
    def test_schema_lists_tables_with_row_counts(self, tool: SQLTool) -> None:
        schema = tool.schema()
        for table in ("stores", "products", "orders", "order_items", "inventory_snapshots"):
            assert f"CREATE TABLE {table}" in schema
        assert "-- 12 rows" in schema

    @pytest.mark.parametrize(
        "statement",
        [
            "DELETE FROM orders",
            "UPDATE stores SET state = 'WA'",
            "INSERT INTO stores VALUES ('X', 'x', 'x', 'x', 'x', 'x', 1)",
            "DROP TABLE orders",
            "CREATE TABLE t (x)",
            "ATTACH DATABASE '/tmp/other.db' AS other",
            "PRAGMA table_info(orders)",
            "SELECT 1; DROP TABLE orders",
            "   ",
        ],
    )
    def test_rejects_anything_but_reads(self, tool: SQLTool, statement: str) -> None:
        with pytest.raises(UnsafeQueryError):
            tool.run(statement)
        assert tool.run("SELECT COUNT(*) FROM stores").rows == [(12,)]  # untouched

    def test_row_cap_and_markdown(self, db_path: Path) -> None:
        result = SQLTool(db_path, max_rows=5).run("SELECT order_id FROM orders;")
        assert len(result.rows) == 5
        assert result.truncated
        rendered = result.to_markdown()
        assert rendered.startswith("| order_id |")
        assert "truncated" in rendered

    def test_runaway_query_times_out(self, db_path: Path) -> None:
        slow = SQLTool(db_path, timeout_seconds=0.05)
        with pytest.raises(TimeoutError):
            slow.run(
                "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) "
                "SELECT COUNT(*) FROM n"
            )

    def test_missing_database(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SQLTool(tmp_path / "missing.db")


class TestRouter:
    @pytest.mark.parametrize(
        ("question", "route"),
        [
            ("How many click and collect orders did HB003 have in July?", "sql"),
            ("What were total sales by state in August 2026?", "sql"),
            ("Which 5 products sold the most units?", "sql"),
            ("How many stores does Harbourline operate in Western Australia?", "sql"),
            ("How quickly must Class A recall stock be removed?", "rag"),
            ("What was Coles' FY25 group EBIT?", "rag"),
            ("How many points does a Gold member earn per dollar?", "rag"),
            ("Which stores meet the shrink escalation threshold in the handbook?", "hybrid"),
            (
                "Which products at HB003 are at risk according to the replenishment policy?",
                "hybrid",
            ),
        ],
    )
    def test_routes(self, question: str, route: str) -> None:
        decision = route_question(question)
        assert decision.route == route, decision.reasons
        assert decision.reasons


class TestSQLAgent:
    def test_returns_none_without_claude(self, tool: SQLTool, settings: Settings) -> None:
        assert answer_with_sql("How many stores?", tool, settings) is None

    def test_tool_loop_executes_sql_and_answers(
        self, monkeypatch: pytest.MonkeyPatch, tool: SQLTool, llm_settings: Settings
    ) -> None:
        fake = install(
            monkeypatch,
            ScriptedAnthropic(
                [
                    message(
                        tool_use_block("SELECT COUNT(*) AS n FROM stores"), stop_reason="tool_use"
                    ),
                    message(
                        text_block("Harbourline has 12 stores (synthetic data)."), cache_read=900
                    ),
                ]
            ),
        )
        result = answer_with_sql("How many stores are there?", tool, llm_settings)
        assert result is not None
        assert result.text.startswith("Harbourline has 12 stores")
        assert not result.refused
        assert result.queries == ["SELECT COUNT(*) AS n FROM stores"]
        assert result.steps == 2
        assert result.usage["input_tokens"] == 200
        assert result.usage["cache_read_input_tokens"] == 900

        first, second = fake.requests
        assert first["tools"][0]["name"] == "run_sql"
        assert first["tools"][0]["strict"] is True
        assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "CREATE TABLE stores" in first["system"][0]["text"]
        assert first["cache_control"] == {"type": "ephemeral"}
        tool_result = second["messages"][-1]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert "| n |" in tool_result["content"]
        assert "12" in tool_result["content"]

    def test_unsafe_sql_is_returned_to_the_model_as_an_error(
        self, monkeypatch: pytest.MonkeyPatch, tool: SQLTool, llm_settings: Settings
    ) -> None:
        fake = install(
            monkeypatch,
            ScriptedAnthropic(
                [
                    message(tool_use_block("DELETE FROM orders"), stop_reason="tool_use"),
                    message(tool_use_block("SELEC oops", "toolu_2"), stop_reason="tool_use"),
                    message(text_block(DATA_REFUSAL_TEXT)),
                ]
            ),
        )
        result = answer_with_sql("Delete all orders", tool, llm_settings)
        assert result is not None
        assert result.refused
        assert result.queries == []  # nothing ran successfully
        rejected = fake.requests[1]["messages"][-1]["content"][0]
        assert rejected["is_error"] is True
        assert rejected["content"].startswith("Rejected:")
        syntax = fake.requests[2]["messages"][-1]["content"][0]
        assert syntax["content"].startswith("SQL error:")
        assert tool.run("SELECT COUNT(*) FROM orders").rows[0][0] > 0

    def test_refusal_stop_reason(
        self, monkeypatch: pytest.MonkeyPatch, tool: SQLTool, llm_settings: Settings
    ) -> None:
        install(monkeypatch, ScriptedAnthropic([message(stop_reason="refusal")]))
        result = answer_with_sql("q", tool, llm_settings)
        assert result is not None
        assert result.refused
        assert result.text == DATA_REFUSAL_TEXT

    def test_step_limit(
        self, monkeypatch: pytest.MonkeyPatch, tool: SQLTool, llm_settings: Settings
    ) -> None:
        looping = [
            message(tool_use_block("SELECT 1", f"toolu_{n}"), stop_reason="tool_use")
            for n in range(3)
        ]
        install(monkeypatch, ScriptedAnthropic(looping))
        result = answer_with_sql("q", tool, llm_settings, max_steps=3)
        assert result is not None
        assert result.refused
        assert result.steps == 3
        assert result.queries == ["SELECT 1"] * 3


class TestPipelineRoutes:
    def _pipeline(self, settings: Settings, tool: SQLTool | None) -> RAGPipeline:
        return RAGPipeline(BM25Retriever(load_chunks(DOCS_DIR)), settings, sql_tool=tool)

    def test_sql_route_without_claude_explains_itself(
        self, settings: Settings, tool: SQLTool
    ) -> None:
        result = self._pipeline(settings, tool).ask("How many orders did HB001 have in July?")
        assert result.route == "sql"
        assert result.answer == SQL_NEEDS_CLAUDE
        assert not result.refused
        assert result.citations == []

    def test_sql_route_falls_back_to_documents_without_a_database(self, settings: Settings) -> None:
        result = self._pipeline(settings, None).ask("How many orders did HB001 have in July?")
        assert result.route == "rag"

    def test_router_can_be_disabled(self, settings: Settings, tool: SQLTool) -> None:
        off = settings.model_copy(update={"router_enabled": False})
        assert self._pipeline(off, tool).ask("How many orders in July?").route == "rag"

    def test_sql_route_with_claude(
        self, monkeypatch: pytest.MonkeyPatch, llm_settings: Settings, tool: SQLTool
    ) -> None:
        install(
            monkeypatch,
            ScriptedAnthropic(
                [
                    message(tool_use_block("SELECT COUNT(*) FROM stores"), stop_reason="tool_use"),
                    message(text_block("12 stores.")),
                ]
            ),
        )
        result = self._pipeline(llm_settings, tool).ask("How many stores are there by state?")
        assert result.route == "sql"
        assert result.generated_by == "claude"
        assert result.sql_queries == ["SELECT COUNT(*) FROM stores"]

    def test_hybrid_route_passes_policy_documents_to_the_sql_agent(
        self, monkeypatch: pytest.MonkeyPatch, llm_settings: Settings, tool: SQLTool
    ) -> None:
        fake = install(monkeypatch, ScriptedAnthropic([message(text_block("HB006."))]))
        question = "Which stores meet the shrink escalation threshold in the handbook?"
        result = self._pipeline(llm_settings, tool).ask(question)
        assert result.route == "hybrid"
        assert result.citations  # the policy chunks are cited
        prompt = fake.requests[0]["messages"][0]["content"]
        assert "<documents>" in prompt
        assert "1.8%" in prompt  # the handbook's threshold reached the SQL agent

    def test_hybrid_without_claude_shows_documents(self, settings: Settings, tool: SQLTool) -> None:
        question = "Which stores meet the shrink escalation threshold in the handbook?"
        result = self._pipeline(settings, tool).ask(question)
        assert result.route == "hybrid"
        assert "Retrieved evidence" in result.answer
        assert SQL_NEEDS_CLAUDE in result.answer

    def test_sql_agent_error_is_not_fatal(
        self, monkeypatch: pytest.MonkeyPatch, llm_settings: Settings, tool: SQLTool
    ) -> None:
        def boom(*_: object, **__: object) -> None:
            raise ConnectionError("network down")

        monkeypatch.setattr("retail_rag.pipeline.answer_with_sql", boom)
        result = self._pipeline(llm_settings, tool).ask("How many orders did HB001 have?")
        assert result.generated_by == "local_fallback"
