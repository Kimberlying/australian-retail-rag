"""Golden-set schema and loader.

Relevance is labelled by *source file + an evidence substring* rather than by
chunk id. Chunk ids change whenever chunking parameters change, which would
silently invalidate the labels; an evidence substring survives re-chunking and
makes chunking itself something the harness can evaluate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

Category = Literal[
    "public_fact",
    "synthetic_policy",
    "paraphrase",
    "multi_doc",
    "unanswerable",
]


class RelevantEvidence(BaseModel):
    source: str
    contains: str | None = Field(
        default=None,
        description="Case-insensitive substring a chunk must contain to count as relevant.",
    )

    def matches(self, source: str, text: str) -> bool:
        if source != self.source:
            return False
        return self.contains is None or self.contains.lower() in text.lower()


class GoldenExample(BaseModel):
    id: str
    question: str = Field(min_length=3)
    category: Category
    answerable: bool = True
    relevant: list[RelevantEvidence] = Field(default_factory=list)
    answer_must_contain: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _check_labels(self) -> GoldenExample:
        if self.answerable and not self.relevant:
            raise ValueError(f"{self.id}: answerable examples need at least one relevant item")
        if not self.answerable and self.answer_must_contain:
            raise ValueError(f"{self.id}: unanswerable examples cannot require answer text")
        return self


def load_golden_set(path: Path, *, docs_dir: Path | None = None) -> list[GoldenExample]:
    """Load a JSONL golden set, validating every row and (optionally) its sources."""
    examples: list[GoldenExample] = []
    errors: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            examples.append(GoldenExample.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"line {line_no}: {exc}")

    ids = [example.id for example in examples]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        errors.append(f"duplicate ids: {', '.join(duplicates)}")

    if docs_dir is not None:
        for example in examples:
            for evidence in example.relevant:
                if not (docs_dir / evidence.source).is_file():
                    errors.append(f"{example.id}: unknown source {evidence.source!r}")

    if errors:
        raise ValueError("Invalid golden set:\n  " + "\n  ".join(errors))
    return examples
