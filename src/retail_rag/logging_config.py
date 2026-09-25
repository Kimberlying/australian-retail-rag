"""Structured logging on top of the standard library.

``LOG_FORMAT=json`` emits one JSON object per line (for log shippers such as
CloudWatch, Datadog, or Loki); ``text`` is the human-friendly default.
Pass structured fields with ``logger.info("msg", extra={"fields": {...}})``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from .config import get_settings

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict) and fields:
            rendered = " ".join(f"{key}={value}" for key, value in fields.items())
            return f"{base} | {rendered}"
        return base


def configure_logging(*, force: bool = False) -> None:
    global _CONFIGURED  # noqa: PLW0603 - process-wide, idempotent setup
    if _CONFIGURED and not force:
        return
    settings = get_settings()
    handler = logging.StreamHandler()
    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("retail_rag")
    root.handlers = [handler]
    root.setLevel(settings.log_level)
    root.propagate = False
    _CONFIGURED = True
