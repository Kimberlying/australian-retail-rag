"""A read-only SQL tool that is safe to hand to an LLM.

The model writes the SQL, so every statement is treated as untrusted input and
defended in depth (any single layer failing still leaves the others):

1. **Read-only connection**: the database is opened with ``mode=ro``, so SQLite
   itself rejects writes.
2. **Authorizer allow-list**: SQLite asks ``_authorize`` about every operation
   while compiling the statement. Only SELECT, column reads, functions, and
   recursive CTEs are allowed; ATTACH, PRAGMA, writes, DDL, and anything else
   are denied before execution starts.
3. **One statement**: ``sqlite3`` refuses to execute more than one statement per
   call, so ``SELECT 1; DROP TABLE orders`` fails outright.
4. **Budgets**: a progress handler aborts long-running queries (an accidental
   cross join), and results are capped at ``max_rows`` so the model's context
   is never flooded.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


class UnsafeQueryError(ValueError):
    """The statement tried to do something other than read."""


@dataclass(frozen=True)
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool

    def to_markdown(self, *, max_rows: int = 50) -> str:
        if not self.columns:
            return "(no columns)"
        lines = [
            "| " + " | ".join(self.columns) + " |",
            "|" + "---|" * len(self.columns),
        ]
        for row in self.rows[:max_rows]:
            lines.append(
                "| " + " | ".join("" if value is None else str(value) for value in row) + " |"
            )
        shown = min(len(self.rows), max_rows)
        footer = f"{len(self.rows)} row(s)"
        if shown < len(self.rows) or self.truncated:
            footer += f"; showing {shown}" + (
                "; result truncated at the row limit" if self.truncated else ""
            )
        return "\n".join([*lines, footer])


def _authorize(action: int, *_: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


class SQLTool:
    def __init__(self, db_path: Path, *, max_rows: int = 200, timeout_seconds: float = 2.0):
        if not db_path.is_file():
            raise FileNotFoundError(f"SQL database not found: {db_path}")
        self.db_path = db_path
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def _connect(self) -> sqlite3.Connection:
        # A fresh connection per call: cheap for SQLite and safe across API threads.
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)

    def schema(self) -> str:
        """``CREATE TABLE`` statements plus row counts, for the model's system prompt."""
        with closing(self._connect()) as conn:
            tables = conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
            parts = []
            for name, ddl in tables:
                (count,) = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()  # noqa: S608 - names from sqlite_master
                parts.append(f"{ddl.strip()};\n-- {count} rows")
        return "\n\n".join(parts)

    def run(self, sql: str) -> QueryResult:
        statement = sql.strip().rstrip(";").strip()
        if not statement:
            raise UnsafeQueryError("empty statement")
        deadline = time.monotonic() + self.timeout_seconds
        with closing(self._connect()) as conn:
            conn.set_authorizer(_authorize)
            # Called every 10k VM instructions; a non-zero return aborts the query.
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
            try:
                cursor = conn.execute(statement)
            except sqlite3.ProgrammingError as exc:  # e.g. more than one statement
                raise UnsafeQueryError(str(exc)) from exc
            except sqlite3.DatabaseError as exc:
                if "not authorized" in str(exc):
                    raise UnsafeQueryError(
                        f"only read-only SELECT queries are allowed ({exc})"
                    ) from exc
                if "interrupted" in str(exc):
                    raise TimeoutError(f"query exceeded {self.timeout_seconds}s") from exc
                raise
            columns = [column[0] for column in cursor.description or []]
            try:
                rows = cursor.fetchmany(self.max_rows + 1)
            except sqlite3.OperationalError as exc:
                if "interrupted" in str(exc):
                    raise TimeoutError(f"query exceeded {self.timeout_seconds}s") from exc
                raise
        return QueryResult(
            sql=statement,
            columns=columns,
            rows=[tuple(row) for row in rows[: self.max_rows]],
            truncated=len(rows) > self.max_rows,
        )
