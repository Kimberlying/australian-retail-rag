"""Held-out routing check.

The router's rules were written while looking at the golden set, so its
``route_accuracy`` there is a training-set number. ``evals/router_holdout.jsonl``
was written after the rules were frozen and is never used to tune them; its
accuracy is the honest estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..router import route_question


def router_holdout(path: Path) -> tuple[float, list[dict[str, str]]]:
    """Return (accuracy, misrouted rows) for a JSONL file of ``{question, route}``."""
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    misses = []
    for row in rows:
        chosen = route_question(row["question"]).route
        if chosen != row["route"]:
            misses.append({"question": row["question"], "expected": row["route"], "chosen": chosen})
    accuracy = 1 - len(misses) / len(rows) if rows else 0.0
    return round(accuracy, 4), misses
