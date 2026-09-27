# Evaluation sets

| File | What it tests | Size |
|---|---|---|
| `golden_set.jsonl` | End-to-end questions: retrieval, refusal, routing, SQL answers | 230 |
| `router_holdout.jsonl` | Routing only; written after the router rules were frozen, never used to tune them | 38 |
| `baselines/` | Committed reports for every retriever, so before/after comparisons stay in the repository | |

## Golden set format

One labelled question per line:

```json
{"id": "rec-003", "question": "How quickly must stock be removed for a Class B recall?",
 "category": "synthetic_policy",
 "relevant": [{"source": "synthetic_product_recall_procedure.md", "contains": "Class B recall, stock must be removed within 24 hours"}],
 "answer_must_contain": ["24 hours"]}

{"id": "sql-002", "question": "How many orders did store HB003 receive in July 2026?",
 "category": "structured_data", "route": "sql",
 "gold_sql": "SELECT COUNT(*) FROM orders o WHERE o.store_id = 'HB003' AND o.order_date BETWEEN '2026-07-01' AND '2026-07-31'"}
```

| Field | Meaning |
|---|---|
| `category` | `public_fact`, `synthetic_policy`, `synthetic_report` (the PDF annual report), `paraphrase` (little word overlap with the source), `multi_doc` (evidence spread over several files), `unanswerable`, `structured_data` (SQL), `hybrid` (policy + data) |
| `relevant` | Evidence a correct retrieval must surface. A chunk counts as relevant when it comes from `source` **and** contains the `contains` substring (case- and whitespace-insensitive, so labels survive PDF line wrapping). |
| `answer_must_contain` | Strings the final answer must include for `answer_accuracy`. |
| `answerable: false` | The system should refuse. Hard negatives reuse corpus vocabulary (*Woolworths* EBIT, *FY24* revenue, Harbourline's *share price*) to test that it does not answer from nearby but wrong evidence. |
| `route` | `rag` (default), `sql`, or `hybrid`: where the router should send the question. |
| `gold_sql` | For `sql` / `hybrid` questions: a reference query. With Claude, the agent's last query must return the same rows (**execution accuracy**; extra columns, row order, and rounding within 0.5% are tolerated). |

Labels use evidence substrings rather than chunk ids, so they survive changes to chunking. That makes chunk size and overlap something the harness can tune (`--chunk-size`, `--chunk-overlap`). `tests/test_evaluation.py` fails if any label stops matching the corpus.

## What each run reports

- Retrieval: hit rate, recall, precision, MRR, nDCG at k (document questions).
- Refusal: refusal accuracy on unanswerable questions, false refusal rate on answerable ones, and which layer refused each question (metadata filter, evidence gate, or model).
- Routing: accuracy and a confusion matrix on the golden set, plus accuracy on the held-out set. The router's rules were written against the golden set, so only the held-out number is an unbiased estimate.
- With `ANTHROPIC_API_KEY`: answer accuracy on generated text, citation validity, SQL execution accuracy, and (with `--judge`) LLM-judged faithfulness.
- An evidence-gate threshold sweep, to recalibrate the gate when the corpus or embedding model changes.

## Adding questions

1. Add a line to `golden_set.jsonl` with a unique `id`.
2. Run `uv run pytest tests/test_evaluation.py` to validate the labels.
3. Run `uv run retail-rag eval` and look at the report under `reports/`.

When you change the retriever, write a new baseline to `baselines/` so the before/after comparison stays in the repository:

```bash
uv run retail-rag eval --output-dir evals/baselines --stem <retriever>_local   # or: make compare
```
