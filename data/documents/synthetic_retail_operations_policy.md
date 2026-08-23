# Synthetic retail operations policy

**Document type:** Synthetic portfolio data — not a real company policy.

## Inventory alert rule

An item is considered at risk when available inventory is below its reorder point for two consecutive daily snapshots. The assistant may explain the rule, but it must not place a purchase order automatically.

## Returns review rule

Returns with an unknown reason code should be routed to a human operations analyst. The assistant may summarise the cases and propose a category, but the final classification requires human review.

## AI response rule

Answers must distinguish public company facts from synthetic operational examples. If a question asks for a metric that is not present in the connected data, the assistant must say that the evidence is insufficient rather than inventing a number.
