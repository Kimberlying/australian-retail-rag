"""Pure, dependency-free IR and generation metrics.

Every function takes plain lists so it can be unit-tested with hand-computed
expectations, independent of any retriever implementation.

``relevance`` is a per-rank list of booleans for the retrieved results (rank 1
first). ``found`` is, per labelled evidence item, whether any retrieved result
matched it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

_CITED_FILE_RE = re.compile(r"[\w./-]+\.(?:md|markdown|txt|pdf)\b", re.IGNORECASE)


def hit_rate(relevance: Sequence[bool]) -> float:
    return 1.0 if any(relevance) else 0.0


def recall(found: Sequence[bool]) -> float:
    return sum(found) / len(found) if found else 0.0


def precision(relevance: Sequence[bool]) -> float:
    return sum(relevance) / len(relevance) if relevance else 0.0


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def ndcg(relevance: Sequence[bool], *, n_relevant: int, k: int) -> float:
    """Binary-relevance nDCG@k. ``n_relevant`` bounds the ideal ranking."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevance[:k], start=1)
        if is_relevant
    )
    ideal_hits = min(n_relevant, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def contains_all(text: str, required: Sequence[str]) -> bool:
    lowered = text.lower()
    return all(item.lower() in lowered for item in required)


def cited_files(text: str) -> set[str]:
    return {match.group(0).rsplit("/", 1)[-1] for match in _CITED_FILE_RE.finditer(text)}


def citation_validity(answer: str, retrieved_sources: Sequence[str]) -> float | None:
    """Share of file names cited in the answer that were actually retrieved.

    Returns ``None`` when the answer cites no files (nothing to validate). A
    value below 1.0 means the model cited a source it was never shown: a
    hallucinated citation.
    """
    cited = cited_files(answer)
    if not cited:
        return None
    retrieved = {source.rsplit("/", 1)[-1] for source in retrieved_sources}
    return len(cited & retrieved) / len(cited)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(math.ceil(pct / 100 * len(ordered)), 1)
    return ordered[rank - 1]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def refusal_sweep(
    answerable: Sequence[float], unanswerable: Sequence[float], thresholds: Sequence[float]
) -> list[dict[str, float]]:
    """Trade-off of an answer-level evidence gate at each candidate threshold.

    Inputs are each question's best retrieval relevance (0 when nothing was
    retrieved). A question is refused when its best relevance is below the
    threshold. Returns refusal accuracy on unanswerable questions and the false
    refusal rate on answerable ones, i.e. the two sides of the operating point.
    """
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "threshold": threshold,
                "refusal_accuracy": mean([float(s < threshold) for s in unanswerable]),
                "false_refusal_rate": mean([float(s < threshold) for s in answerable]),
            }
        )
    return rows


def _cell_matches(gold: object, predicted: object) -> bool:
    numeric = (int, float)
    if isinstance(gold, numeric) and isinstance(predicted, numeric):
        # Tolerate rounding choices: 0.5% relative, or 0.01 absolute for small values.
        return abs(float(gold) - float(predicted)) <= max(0.01, 0.005 * abs(float(gold)))
    return str(gold).strip().lower() == str(predicted).strip().lower()


def _row_covers(predicted: Sequence[object], gold: Sequence[object]) -> bool:
    """True if every gold value matches a distinct value in the predicted row."""
    unused = list(predicted)
    for value in gold:
        match = next((i for i, cell in enumerate(unused) if _cell_matches(value, cell)), None)
        if match is None:
            return False
        del unused[match]
    return True


def result_matches(gold: Sequence[Sequence[object]], predicted: Sequence[Sequence[object]]) -> bool:
    """Execution accuracy with column and rounding tolerance.

    The predicted result must have the same number of rows, and every gold row
    must be covered by a distinct predicted row that contains all of its values.
    Extra predicted columns (a store name next to the store id) are allowed, row
    order is ignored, and numbers match within 0.5% so ``ROUND(x, 1)`` versus an
    unrounded gold value is not counted as wrong.
    """
    if len(gold) != len(predicted):
        return False
    remaining = list(predicted)
    for gold_row in gold:
        match = next((i for i, row in enumerate(remaining) if _row_covers(row, gold_row)), None)
        if match is None:
            return False
        del remaining[match]
    return True
