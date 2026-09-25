# Australian Retail RAG

[![CI](https://github.com/kimberlying/australian-retail-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/kimberlying/australian-retail-rag/actions/workflows/ci.yml)

A retrieval-augmented generation (RAG) system for Australian retail, built around evaluation. It uses public Australian company material alongside clearly labelled synthetic retail data.

It is an independent portfolio project. It is **not a Coles system, customer project, or endorsement**. "Harbourline Retail" in the synthetic documents is a fictional company.

## Highlights

- **Evaluation harness:** a 55-question golden set with retrieval metrics (Hit@k, Recall@k, MRR, nDCG, Precision@k) and generation metrics (answer accuracy, refusal accuracy, citation validity, and LLM-as-judge faithfulness). CI fails when retrieval quality drops.
- **Grounded generation with Claude:** evidence is passed to the model as XML, which helps resist prompt injection. Answers include citations, and refusals are machine-detectable. Server-side refusal fallback is enabled, and the system degrades to a local retrieval preview when there is no API key or the API call fails.
- **Production engineering:** typed settings (pydantic-settings), structured JSON logs with request ids, a FastAPI service that loads the index once at startup with separate liveness and readiness probes, a multi-stage non-root Docker image, a `uv` lockfile, ruff, strict mypy, pytest with coverage, pre-commit, and GitHub Actions.

## Architecture

```text
 data/documents/*.md|pdf ──► ingest ──► chunk (size/overlap) ──► TF-IDF index
                                                                     │
 question ──► retrieve top-k ──► min-score gate ──┬── no evidence ──► refuse (no LLM call)
                                                  │
                                                  └── evidence ──► Claude (XML-wrapped context)
                                                                     │   └── error ─► local preview
                                                                     ▼
                                                   answer + citations + latency + token usage

 evals/golden_set.jsonl ──► retail-rag eval ──► metrics ──► reports/eval.{md,json} ──► CI gate
```

## Quick start

```bash
make install          # uv sync --all-extras + pre-commit hooks
make ingest           # build data/index/index.json
uv run retail-rag query "What was Coles' normalised eCommerce sales growth in FY25?"
```

Without `ANTHROPIC_API_KEY` the query returns a local retrieval preview. To get grounded Claude answers:

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY; never commit .env
uv run retail-rag query "What was Coles' normalised eCommerce sales growth in FY25?" --json
```

### API

```bash
make serve                                   # or: docker compose up --build
curl localhost:8000/health                   # liveness
curl localhost:8000/ready                    # readiness (index loaded?)
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"How quickly must Class A recall stock be removed?"}'
```

The `/query` response includes `answer`, `citations`, `generated_by` (`claude` / `local` / `local_fallback`), `refused`, and `latency_ms`. Every response carries an `x-request-id` header, and the same id appears in the logs.

## Evaluation

```bash
make eval                                    # same gate CI runs
uv run retail-rag eval --k 6 --chunk-size 600 --stem k6_c600   # experiment
uv run retail-rag eval --judge --stem claude  # with ANTHROPIC_API_KEY: generation + LLM judge
```

| Metric | What it measures |
|---|---|
| Hit rate@k / Recall@k | Did the right evidence reach the model at all? |
| MRR / nDCG@k | Was the right evidence ranked near the top? |
| Answer accuracy | Does the answer contain the expected facts? |
| Refusal accuracy | Does the system decline questions the corpus cannot answer? |
| Citation validity | Does every file the answer cites appear among the retrieved chunks? Anything else is a hallucinated citation. |
| Faithfulness | LLM-as-judge (structured output): is every claim supported by the evidence? |

The golden set format is documented in [`evals/README.md`](evals/README.md). Baseline reports are committed in [`evals/baselines/`](evals/baselines/).

### Current baseline: TF-IDF, local mode

| | Hit@4 | Recall@4 | MRR | Refusal accuracy |
|---|---|---|---|---|
| Literal questions (public + policy) | 1.00 | 1.00 | 0.97 | – |
| Paraphrased questions | 0.78 | 0.72 | 0.70 | – |
| Unanswerable (11) | – | – | – | **0.00** |

What the baseline shows, and what to build next:

1. **Lexical retrieval breaks on paraphrases.** "freezer" does not match "frozen", and "stealing" does not match "shoplifter". This is the motivation for dense embeddings plus hybrid BM25 and vector search.
2. **A score threshold cannot provide refusal with TF-IDF.** "What is the capital of France?" scores 0.335, which is higher than several correctly answered questions. When a query shares only stopwords with the vocabulary, its cosine score is inflated. Refusal therefore has to come from the generator (the prompt contract plus `refusal_accuracy` measured in Claude mode), from better retrieval scores, or from a reranker.

## Engineering

| Concern | Implementation |
|---|---|
| Dependencies | `uv` + committed `uv.lock`; optional extras `api`, `llm`, `pdf`, `dev` |
| Config | `pydantic-settings` with validation (`src/retail_rag/config.py`); secrets are held as `SecretStr` |
| Quality | `ruff` (lint + format + bandit rules), `mypy --strict`, `pre-commit` |
| Tests | 65 pytest tests covering unit, API, CLI, and eval logic; Claude is faked so tests are offline and deterministic; the CI coverage gate is 85% |
| CI | lint → tests (py3.11/3.12) → eval gate (report shown in the job summary) → Docker build + smoke test; manual live-Claude eval job |
| Container | multi-stage build, non-root user, index baked in, `HEALTHCHECK` on `/ready`, JSON logs |
| Observability | structured logs with `request_id`, per-stage latency, token usage, `generated_by`, `refused` |

```bash
make check   # lint + typecheck + tests + eval gate — everything CI runs
```

## Repository layout

```text
src/retail_rag/
  config.py          typed settings from env / .env
  logging_config.py  text or JSON structured logging
  chunking.py        deterministic chunking
  retrieval.py       TF-IDF retriever + persistence
  ingest.py          markdown/text/PDF ingestion
  generation.py      Claude generation (XML context, refusal contract, fallback)
  pipeline.py        orchestration, score gate, latency/usage tracking
  api.py             FastAPI service (lifespan-loaded index, request ids, probes)
  cli.py             ingest / query / serve / eval
  evaluation/        golden-set schema, metrics, runner, reports, LLM judge
evals/               golden set + committed baseline reports
data/documents/      public snapshot + synthetic documents
tests/               pytest suite
```

## Data and claim boundaries

- The Coles snapshot links back to the official source page. Refresh it before a public demo.
- All Harbourline documents are synthetic. Transactional data must stay synthetic unless there is explicit permission to use real data.
- Do not describe this repository as a Coles deployment or customer engagement.
- Do not commit API keys, private documents, personal data, or former-employer data.

## Roadmap

1. ~~Evaluation harness with golden set and CI quality gate~~ ✅
2. ~~Engineering baseline: uv, ruff, mypy, Docker, CI, structured logging~~ ✅
3. Dense embeddings + pgvector, hybrid BM25/vector retrieval with RRF, and a reranker, compared against the TF-IDF baseline
4. Layout-aware PDF parsing of real annual reports, with company, year, and page metadata filters
5. Synthetic orders and inventory in Postgres, a read-only SQL tool, and a RAG / SQL / hybrid router
6. Tracing (Langfuse/OpenTelemetry), prompt caching, streaming responses, auth and rate limiting
