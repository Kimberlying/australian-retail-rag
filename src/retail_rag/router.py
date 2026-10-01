"""Route a question to documents (RAG), the operational database (SQL), or both.

A rule-based router is the right first version here: it is instant, free,
deterministic, and every decision comes with the rules that fired, so a wrong
route is debuggable. Its accuracy is measured on labelled questions in the
evaluation (``route_accuracy``); an LLM classifier is the upgrade path if that
number stops being good enough.

The rules encode what distinguishes the two sources:

* **SQL** questions ask for a *measurement over operational records*: an
  aggregate or ranking ("how many", "total", "top 5", "which store had the
  most") over data nouns (orders, units, stock on hand, shrink, stores), often
  scoped to a store id, state, channel, or month.
* **Document** questions ask what a *rule, policy, or public report* says:
  "must", "policy", "how quickly", "what happens", or a public company's results.
* **Hybrid** questions need a policy definition applied to the data ("which
  stores meet the escalation threshold in the handbook?").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Route

_AGGREGATE = re.compile(
    r"\b(how many|how much|total|sum|average|avg|mean|count|number of|top \d+|top|"
    r"highest|lowest|most|least|best[- ]selling|rank|ranking|share of|percentage of|"
    r"proportion of|compare|trend|median|busiest|sold)\b",
    re.IGNORECASE,
)
# Nouns that only make sense as rows in the operational database.
_STRONG_DATA_NOUN = re.compile(
    r"\b(orders?|units|units sold|line items?|skus?|stock on hand|on hand|snapshots?|"
    r"baskets?|picking times?|order value)\b",
    re.IGNORECASE,
)
# Nouns that appear in both the data and the documents ("products", "sales").
_WEAK_DATA_NOUN = re.compile(
    r"\b(sold|sales|revenue|takings|inventory|stock levels?|shrink|stores?|products?|"
    r"categor(y|ies)|channels?|customers?)\b",
    re.IGNORECASE,
)
_DATA_SCOPE = re.compile(
    r"\b(HB\d{3}|SKU-\d{4}|NSW|VIC|QLD|SA|ACT|TAS|WA|NT|Western Australia|Queensland|"
    r"Victoria|New South Wales|Tasmania|South Australia|June|July|August|September|2026|"
    r"per store|by store|by state|each state|per state|by channel|per channel|"
    r"per sales channel|by category|each store|which stores?|last (week|month|quarter))\b",
    re.IGNORECASE,
)
_DOC_SIGNAL = re.compile(
    r"\b(policy|policies|procedure|handbook|terms|rule|rules|must|should|allowed|may|"
    r"required?|requirement|responsible|escalat\w*|sign[- ]off|approv\w*|how quickly|how fast|"
    r"how long|how often|what happens|what should|why|explain|define|definition|mean by|means|"
    r"calculated|official|source|endorsed|expire|earn|eligible|points|per dollar|"
    r"safety stock|shelf life|marked down|markdown|lead time|window|cycle-counted|"
    r"promotions?|label|fee|dividend|annual report|outlook|risks?)\b",
    re.IGNORECASE,
)
# The database holds June-August 2026 by date; fiscal-year figures come from reports.
_FISCAL_YEAR = re.compile(r"\bFY\s?'?\d{2,4}\b", re.IGNORECASE)
_PUBLIC_COMPANY = re.compile(
    r"\b(coles|woolworths|wesfarmers|aldi|metcash|jb hi-fi|harvey norman|myer|kmart|bunnings)\b",
    re.IGNORECASE,
)
_APPLY_POLICY = re.compile(
    r"\b(according to|under the|per the|as defined|meets?|breach(es|ed)?|exceed(s|ed)?|"
    r"threshold|at risk|should be escalated|missed the|within the)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    sql_score: int
    doc_score: int
    reasons: list[str] = field(default_factory=list)


def route_question(question: str) -> RouteDecision:
    reasons: list[str] = []
    sql_score = doc_score = 0

    aggregate = _AGGREGATE.search(question)
    strong = _STRONG_DATA_NOUN.search(question)
    weak = _WEAK_DATA_NOUN.search(question)
    noun = strong or weak
    if aggregate and noun:
        sql_score += 3 if strong else 2
        reasons.append(f"aggregate '{aggregate.group(0)}' over '{noun.group(0)}'")
    elif noun:
        sql_score += 1
        reasons.append(f"data noun '{noun.group(0)}'")
    scopes = sorted({match.group(0).lower() for match in _DATA_SCOPE.finditer(question)})
    if scopes:
        sql_score += 2 + min(len(scopes) - 1, 1)
        reasons.append(f"data scope {scopes}")

    doc_hits = {match.group(0).lower() for match in _DOC_SIGNAL.finditer(question)}
    if doc_hits:
        doc_score += 2 + min(len(doc_hits) - 1, 2)
        reasons.append(f"document signal {sorted(doc_hits)}")
    if fiscal := _FISCAL_YEAR.search(question):
        doc_score += 3
        reasons.append(f"fiscal-year figure '{fiscal.group(0)}' (reported in documents)")
    if company := _PUBLIC_COMPANY.search(question):
        # Public companies' figures live in their reports, not in Harbourline's database.
        doc_score += 5
        reasons.append(f"public company '{company.group(0)}'")

    apply_policy = _APPLY_POLICY.search(question)
    if sql_score >= 3 and doc_score >= 2 and apply_policy:
        reasons.append(f"applies a policy to data '{apply_policy.group(0)}'")
        return RouteDecision("hybrid", sql_score, doc_score, reasons)
    if sql_score >= 3 and sql_score > doc_score:
        return RouteDecision("sql", sql_score, doc_score, reasons)
    return RouteDecision("rag", sql_score, doc_score, reasons or ["default: documents"])
