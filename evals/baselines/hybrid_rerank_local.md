# RAG evaluation report

> Generated in CI (run 36290784571, job "Retrieval quality gate"), because the
> cross-encoder is downloaded from Hugging Face at runtime. The JSON version is
> in that run's `eval-report` artifact. Reproduce with
> `uv run retail-rag eval --retriever hybrid --reranker cross-encoder`.

- **Created:** 2026-09-27T03:15:28+00:00
- **Git SHA:** `788a3f7` · **version:** 0.4.0
- **Generation mode:** local
- **Config:** `k=4`, `retriever=hybrid+rerank`, `embedding_model=BAAI/bge-small-en-v1.5`, `chunk_size=900`, `chunk_overlap=120`, `reranker=Xenova/ms-marco-MiniLM-L-6-v2`, `metadata_filters=True`, `router=True`, `min_score=0.575`, `n_chunks=31`, `model=None`

> Generation ran in local-preview mode (no `ANTHROPIC_API_KEY`), so answer
> metrics reflect retrieved evidence only and refusals come solely from the
> retrieval score threshold.

## Summary

| Metric | Value |
|---|---|
| Hit rate@k | 0.994 |
| Recall@k | 0.986 |
| MRR | 0.940 |
| nDCG@k | 0.945 |
| Precision@k | 0.325 |
| Answer accuracy | 0.994 |
| Refusal accuracy (unanswerable) | 0.500 |
| False refusal rate (answerable) | 0.000 |
| Route accuracy (RAG / SQL / hybrid) | 0.996 |
| Route accuracy, held-out set | 0.947 |
| Overall pass rate | 0.935 |
| Latency p50 (ms) | 458.160 |
| Latency p95 (ms) | 513.440 |

## By category

| Category | n_examples | hit_rate | recall | mrr | refusal_accuracy | pass_rate |
|---|---|---|---|---|---|---|
| hybrid | 7 | 1.000 | 1.000 | 0.857 | - | 0.857 |
| multi_doc | 12 | 1.000 | 0.917 | 1.000 | - | 1.000 |
| paraphrase | 26 | 0.962 | 0.942 | 0.801 | - | 0.962 |
| public_fact | 7 | 1.000 | 1.000 | 1.000 | - | 1.000 |
| structured_data | 24 | - | - | - | - | 1.000 |
| synthetic_policy | 110 | 1.000 | 1.000 | 0.977 | - | 1.000 |
| synthetic_report | 18 | 1.000 | 1.000 | 0.880 | - | 1.000 |
| unanswerable | 26 | - | - | - | 0.500 | 0.500 |

## Refusal layers (unanswerable questions)

| Layer | Questions |
|---|---|
| metadata_filter | 7 |
| evidence_gate | 6 |
| answered (missed) | 13 |

## Routing

Rows are the labelled route, columns the route the router chose.

| Expected \ Chosen | rag | sql | hybrid |
|---|---|---|---|
| rag | 199 | 0 | 0 |
| sql | 0 | 24 | 0 |
| hybrid | 1 | 0 | 6 |

The router's rules were written against the golden set, so the table above is a
training-set number. On the held-out set (`evals/router_holdout.jsonl`, never used
for tuning) accuracy is **0.947**.
- held-out miss: expected sql, chose rag: _Which regional store had the highest revenue over the whole period?_
- held-out miss: expected hybrid, chose rag: _Using the inventory alert rule, list at-risk SKUs at HB009._

## Evidence-gate threshold sweep

Best-chunk relevance per question, measured before the gate. A question is
refused when it falls below the threshold (this run: `0.575`). Questions
already refused by the metadata filter never reach the gate and are excluded.

- Answerable: min 0.577, median 0.756
- Unanswerable: max 0.801, median 0.679

| Threshold | Refusal accuracy | False refusal rate |
|---|---|---|
| 0.00 | 0.000 | 0.000 |
| 0.10 | 0.000 | 0.000 |
| 0.20 | 0.000 | 0.000 |
| 0.30 | 0.000 | 0.000 |
| 0.40 | 0.000 | 0.000 |
| 0.50 | 0.158 | 0.000 |
| 0.55 | 0.263 | 0.000 |
| 0.60 | 0.316 | 0.012 |
| 0.65 | 0.368 | 0.087 |
| 0.70 | 0.684 | 0.237 |
| 0.75 | 0.789 | 0.486 |
| 0.80 | 0.947 | 0.786 |

## Failures (15)

- **para-004** (paraphrase) — relevant evidence not retrieved; answer missing expected facts. Top hit: `synthetic_workplace_health_safety.md`. Q: _What should a team member do if they see someone stealing?_
- **un-005** (unanswerable) — should have refused. Top hit: `harbourline_fy25_annual_report.pdf`. Q: _How many Gold tier members does Harbourline Rewards have?_
- **un-006** (unanswerable) — should have refused. Top hit: `synthetic_product_recall_procedure.md`. Q: _What is the return window for change-of-mind returns at Harbourline?_
- **un-008** (unanswerable) — should have refused. Top hit: `harbourline_fy25_annual_report.pdf`. Q: _What is the current Reserve Bank of Australia cash rate?_
- **un-012** (unanswerable) — should have refused. Top hit: `synthetic_supplier_code_of_conduct.md`. Q: _What was Harbourline's FY23 sales revenue?_
- **un-014** (unanswerable) — should have refused. Top hit: `harbourline_fy25_annual_report.pdf`. Q: _What is Harbourline's share price?_
- **un-016** (unanswerable) — should have refused. Top hit: `harbourline_fy25_annual_report.pdf`. Q: _What is the minimum wage for retail workers in Australia?_
- **un-017** (unanswerable) — should have refused. Top hit: `harbourline_fy25_annual_report.pdf`. Q: _How many car parking spaces does the Parramatta store have?_
- **un-019** (unanswerable) — should have refused. Top hit: `harbourline_fy25_annual_report.pdf`. Q: _What is the name of Harbourline's chief executive?_
- **un-020** (unanswerable) — should have refused. Top hit: `harbourline_fy25_annual_report.pdf`. Q: _What staff discount do Harbourline team members get?_
- **un-021** (unanswerable) — should have refused. Top hit: `synthetic_supplier_code_of_conduct.md`. Q: _How much does Harbourline charge for reusable shopping bags?_
- **un-024** (unanswerable) — should have refused. Top hit: `harbourline_fy25_annual_report.pdf`. Q: _How much revenue did Harbourline make from home delivery alone in FY25?_
- **un-025** (unanswerable) — should have refused. Top hit: `harbourline_fy25_annual_report.pdf`. Q: _What penalty rate applies to Sunday shifts at Harbourline?_
- **un-028** (unanswerable) — should have refused. Top hit: `synthetic_product_recall_procedure.md`. Q: _How long is the warranty on electrical products sold by Harbourline?_
- **hyb-003** (hybrid) — routed to rag, expected hybrid. Top hit: `synthetic_retail_operations_policy.md`. Q: _How many store and product pairs are at risk under the inventory alert rule?_
