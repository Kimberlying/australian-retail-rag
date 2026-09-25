# RAG evaluation report

- **Created:** 2026-09-25T03:40:18+00:00
- **Git SHA:** `cc85b3b` · **version:** 0.3.0
- **Generation mode:** local
- **Config:** `k=4`, `retriever=hybrid`, `embedding_model=BAAI/bge-small-en-v1.5`, `chunk_size=900`, `chunk_overlap=120`, `min_score=0.52`, `n_chunks=15`, `model=None`

> Generation ran in local-preview mode (no `ANTHROPIC_API_KEY`), so answer
> metrics reflect retrieved evidence only and refusals come solely from the
> retrieval score threshold.

## Summary

| Metric | Value |
|---|---|
| Hit rate@k | 0.977 |
| Recall@k | 0.955 |
| MRR | 0.932 |
| nDCG@k | 0.969 |
| Precision@k | 0.284 |
| Answer accuracy | 0.973 |
| Refusal accuracy (unanswerable) | 0.273 |
| False refusal rate (answerable) | 0.000 |
| Overall pass rate | 0.836 |
| Latency p50 (ms) | 2.490 |
| Latency p95 (ms) | 3.430 |

## By category

| Category | n_examples | hit_rate | recall | mrr | refusal_accuracy | pass_rate |
|---|---|---|---|---|---|---|
| multi_doc | 4 | 1.000 | 0.875 | 1.000 | - | 1.000 |
| paraphrase | 9 | 0.889 | 0.833 | 0.833 | - | 0.889 |
| public_fact | 7 | 1.000 | 1.000 | 1.000 | - | 1.000 |
| synthetic_policy | 24 | 1.000 | 1.000 | 0.938 | - | 1.000 |
| unanswerable | 11 | - | - | - | 0.273 | 0.273 |

## Evidence-gate threshold sweep

Best-chunk relevance per question, measured before the gate. A question is
refused when it falls below the threshold (this run: `0.52`).

- Answerable: min 0.562, median 0.771
- Unanswerable: max 0.777, median 0.672

| Threshold | Refusal accuracy | False refusal rate |
|---|---|---|
| 0.00 | 0.000 | 0.000 |
| 0.10 | 0.000 | 0.000 |
| 0.20 | 0.000 | 0.000 |
| 0.30 | 0.000 | 0.000 |
| 0.40 | 0.000 | 0.000 |
| 0.50 | 0.273 | 0.000 |
| 0.55 | 0.273 | 0.000 |
| 0.60 | 0.273 | 0.068 |
| 0.65 | 0.364 | 0.091 |
| 0.70 | 0.636 | 0.227 |
| 0.75 | 0.818 | 0.341 |
| 0.80 | 1.000 | 0.795 |

## Failures (9)

- **para-004** (paraphrase) — relevant evidence not retrieved; answer missing expected facts. Top hit: `synthetic_product_recall_procedure.md`. Q: _What should a team member do if they see someone stealing?_
- **un-001** (unanswerable) — should have refused. Top hit: `coles_fy25_public_snapshot.md`. Q: _What was Woolworths' FY25 group EBIT?_
- **un-002** (unanswerable) — should have refused. Top hit: `coles_fy25_public_snapshot.md`. Q: _What was Coles' FY24 group sales revenue?_
- **un-003** (unanswerable) — should have refused. Top hit: `coles_fy25_public_snapshot.md`. Q: _What was Coles' normalised eCommerce sales growth in FY23?_
- **un-004** (unanswerable) — should have refused. Top hit: `synthetic_store_operations_handbook.md`. Q: _What is Harbourline Retail's annual revenue?_
- **un-005** (unanswerable) — should have refused. Top hit: `synthetic_loyalty_program_terms.md`. Q: _How many Gold tier members does Harbourline Rewards have?_
- **un-006** (unanswerable) — should have refused. Top hit: `synthetic_inventory_replenishment_policy.md`. Q: _What is the return window for change-of-mind returns at Harbourline?_
- **un-007** (unanswerable) — should have refused. Top hit: `synthetic_store_operations_handbook.md`. Q: _How many stores does Harbourline operate in Western Australia?_
- **un-008** (unanswerable) — should have refused. Top hit: `synthetic_inventory_replenishment_policy.md`. Q: _What is the current Reserve Bank of Australia cash rate?_
