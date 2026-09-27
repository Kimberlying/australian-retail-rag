"""Structured-data route: a synthetic operational database and a read-only SQL tool."""

from .synthetic import build_database
from .tool import QueryResult, SQLTool, UnsafeQueryError

__all__ = ["QueryResult", "SQLTool", "UnsafeQueryError", "build_database"]
