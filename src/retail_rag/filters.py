"""Self-query metadata filters: read the company and fiscal year a question names.

A similarity threshold cannot tell "Coles FY25 revenue" from "Coles FY24
revenue": both are on-topic and embed almost identically. But the question
states the period explicitly, and every document says which period it covers.
Turning those explicit mentions into a metadata filter makes wrong-period and
wrong-company questions retrieve nothing, so they are refused deterministically
before any LLM call.

The two keys filter differently. ``company`` is strict: a question about
Woolworths may only use Woolworths documents. ``fiscal_year`` excludes documents
dated to *other* years but keeps undated ones, because policies are timeless: "how
did the FY25 out-of-stock rate compare with the policy target?" needs both the
FY25 report and the undated replenishment policy.

The extraction is deliberately conservative. Only explicit fiscal-year forms
(``FY25``, ``FY2025``, ``2024-25``, ``financial year 2025``) are read; a bare
calendar year or "last financial year" adds no filter, because guessing a
period and filtering on the guess would cause false refusals.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

# Australian retailers a user might ask about. A company that is named but absent
# from the corpus is exactly the case the filter exists for ("Woolworths EBIT").
KNOWN_COMPANIES: dict[str, str] = {
    "coles": "Coles",
    "woolworths": "Woolworths",
    "wesfarmers": "Wesfarmers",
    "bunnings": "Bunnings",
    "kmart": "Kmart",
    "officeworks": "Officeworks",
    "big w": "Big W",
    "aldi": "Aldi",
    "metcash": "Metcash",
    "iga": "IGA",
    "jb hi-fi": "JB Hi-Fi",
    "harvey norman": "Harvey Norman",
    "myer": "Myer",
    "david jones": "David Jones",
    "endeavour group": "Endeavour Group",
    "harbourline": "Harbourline Retail",
}

_FY_PATTERNS = [
    re.compile(r"\bFY\s?'?(\d{4}|\d{2})\b", re.IGNORECASE),
    re.compile(r"\b(?:fiscal|financial)\s+year\s+(?:20)?(\d{2})\b", re.IGNORECASE),
    # Australian financial years run July-June: "2024-25" / "2024/25" is FY25.
    re.compile(r"\b20\d{2}\s?[-/]\s?(?:20)?(\d{2})\b"),
]


def _to_year(digits: str) -> int:
    return int(digits) if len(digits) == 4 else 2000 + int(digits)


def fiscal_years(question: str) -> list[int]:
    years = {_to_year(match) for pattern in _FY_PATTERNS for match in pattern.findall(question)}
    return sorted(years)


def companies(question: str, corpus_companies: Iterable[str] = ()) -> list[str]:
    aliases = dict(KNOWN_COMPANIES)
    aliases.update({name.lower(): name for name in corpus_companies})
    found = set()
    lowered = question.lower()
    for alias, canonical in aliases.items():
        # Word boundaries, and allow a possessive ("Coles'", "Harbourline's").
        if re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", lowered):
            found.add(canonical)
    return sorted(found)


def extract_filters(question: str, corpus_companies: Iterable[str] = ()) -> dict[str, list[Any]]:
    """Return a ``MetadataFilter`` for the entities the question names explicitly."""
    where: dict[str, list[Any]] = {}
    if named := companies(question, corpus_companies):
        where["company"] = list(named)
    if years := fiscal_years(question):
        # None: undated documents (policies, procedures) are timeless and stay eligible;
        # only documents about *other* periods are excluded.
        where["fiscal_year"] = [*years, None]
    return where


def corpus_companies(metadata: Iterable[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in metadata:
        value = item.get("company")
        names.update(value if isinstance(value, list) else [value] if value else [])
    return names


def describe(where: Mapping[str, Iterable[Any]]) -> str:
    parts = []
    if "company" in where:
        parts.append(" / ".join(str(value) for value in where["company"]))
    if "fiscal_year" in where:
        years = [value for value in where["fiscal_year"] if value is not None]
        parts.append(" / ".join(f"FY{int(value) % 100:02d}" for value in years))
    return " ".join(parts)
