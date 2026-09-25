# Evaluation golden set

`golden_set.jsonl` holds one labelled question per line:

```json
{"id": "pol-013", "question": "How quickly must stock be removed from shelves for a Class A recall?",
 "category": "synthetic_policy", "answerable": true,
 "relevant": [{"source": "synthetic_product_recall_procedure.md", "contains": "within 2 hours of the recall notice"}],
 "answer_must_contain": ["2 hours"]}
```

| Field | Meaning |
|---|---|
| `category` | `public_fact`, `synthetic_policy`, `paraphrase` (little word overlap with the source), `multi_doc` (evidence spread over several files), `unanswerable` |
| `relevant` | The evidence a correct retrieval must surface. A chunk counts as relevant when it comes from `source` **and** contains the `contains` substring (case-insensitive). |
| `answer_must_contain` | Strings the final answer must include for `answer_accuracy`. |
| `answerable: false` | The system should refuse. Hard negatives reuse corpus vocabulary (e.g. *Woolworths* EBIT, *FY24* revenue) to test that it does not answer from nearby but wrong evidence. |

Labels use evidence substrings rather than chunk ids, so they survive changes to chunking. That makes chunk size and overlap something the harness can tune (`--chunk-size`, `--chunk-overlap`). `tests/test_evaluation.py` fails if any label stops matching the corpus.

## Adding questions

1. Add a line to `golden_set.jsonl` with a unique `id`.
2. Run `uv run pytest tests/test_evaluation.py` to validate the labels.
3. Run `uv run retail-rag eval` and look at the report under `reports/`.

When you change the retriever, write a new baseline to `baselines/` so the before/after comparison stays in the repository:

```bash
uv run retail-rag eval --output-dir evals/baselines --stem <retriever>_local   # or: make compare
```
