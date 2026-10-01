# Australian Retail RAG

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-lightgrey.svg)](README.zh-CN.md)
[![CI](https://github.com/kimberlying/australian-retail-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/kimberlying/australian-retail-rag/actions/workflows/ci.yml)

A question-answering system for Australian retail, built around evaluation. It answers from documents (retrieval-augmented generation) and from an operational database (text-to-SQL), and routes each question to the right source. It uses public Australian company material alongside clearly labelled synthetic retail data.

It is an independent portfolio project. It is **not a Coles system, customer project, or endorsement**. "Harbourline Retail" in the synthetic documents and data is a fictional company.

## Highlights

- **Evaluation first:** a 230-question golden set covering retrieval, refusal, routing, and SQL answers, plus a 38-question held-out routing set. Every design choice below comes with a measured before/after, and CI fails when quality drops.
- **Hybrid retrieval with optional reranking:** BM25 and dense embeddings (`bge-small-en-v1.5`, local ONNX) fused with Reciprocal Rank Fusion, and an optional cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`). The dense side runs in memory or in **Postgres + pgvector (HNSW)** behind the same interface.
- **Layered refusal:** self-query **metadata filters** refuse wrong-company and wrong-period questions ("Woolworths EBIT", "Coles FY24 revenue") with no LLM call. A calibrated **evidence gate** refuses off-topic questions. The **generator's refusal contract** handles the rest. Each refusal records which layer fired.
- **Documents and data:** a rule-based router sends each question to documents, to a seeded **synthetic operational database** (orders, inventory, shrink), or to both. Claude writes the SQL in a tool-use loop against a **read-only SQL tool** with defence in depth.
- **Page-aware PDF ingestion:** a synthetic FY25 annual report is parsed page by page, with running headers and footers stripped. Citations carry page numbers, and company and fiscal-year metadata come from front matter or sidecar files.
- **Production serving:** streaming answers over server-sent events, prompt caching on stable prefixes, API-key auth with constant-time comparison, per-client rate limiting, OpenTelemetry tracing with GenAI attributes (works with Langfuse), structured JSON logs, and liveness and readiness probes.
- **Engineering baseline:** `uv` lockfile, ruff, strict mypy, 186 tests at 92% coverage (including real-model and real-Postgres integration tests in CI), a multi-stage non-root Docker image, and GitHub Actions.

## Architecture

```text
 data/documents/*.md|pdf ─► ingest ─► pages ─► chunks (+ company, fiscal_year, page)
                              │                     │
                              │                     ├─► BM25 index (in memory)
                              │                     └─► embeddings: numpy (default) or pgvector HNSW
                              └─► seeded synthetic SQLite: stores, products, orders, inventory, shrink

 question ─► router ─┬─ rag ────► metadata filter ─► hybrid retrieval (BM25 + dense, RRF)
                     │                 │ no match           └─► optional cross-encoder rerank
                     │                 ▼                                   │
                     │           refuse (no LLM)      evidence gate: best similarity < 0.575 ?
                     │                                      │ yes: refuse (no LLM)   │ no
                     │                                      ▼                        ▼
                     │                                                 Claude (XML evidence, streamed)
                     ├─ sql ────► Claude tool-use loop ─► run_sql (read-only) ─► answer + SQL shown
                     └─ hybrid ─► retrieve the policy ─► SQL agent applies it to the data

 every stage ─► OpenTelemetry spans + JSON logs (request id, route, refusal reason, tokens, latency)
 evals/*.jsonl ─► retail-rag eval ─► metrics + reports ─► CI quality gate
```

## Quick start

```bash
make install          # uv sync --all-extras + pre-commit hooks
make ingest           # document index + synthetic operational database
uv run retail-rag query "What was Coles' normalised eCommerce sales growth in FY25?"
uv run retail-rag query "How quickly must Class A recall stock be removed?" --stream
```

The first `ingest` downloads the embedding model (about 70 MB) and caches the chunk embeddings next to the index. For a fully offline run, set `RAG_RETRIEVER=bm25`; to use a pre-downloaded model directory, set `RAG_EMBEDDING_MODEL_PATH`.

Without `ANTHROPIC_API_KEY`, document questions return a local retrieval preview, and database questions explain that Claude is needed to write the SQL. To get Claude answers:

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY; never commit .env
uv run retail-rag query "Which store had the highest sales revenue in July 2026?" --json
```

### API

```bash
make serve                                   # or: docker compose up --build
curl localhost:8000/health                   # liveness
curl localhost:8000/ready                    # readiness: index, SQL, auth, tracing status
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -H 'x-api-key: <key>' -d '{"question":"How quickly must Class A recall stock be removed?"}'
curl -N -X POST localhost:8000/query/stream -H 'content-type: application/json' \
  -d '{"question":"What happens to chilled stock after a cold chain breach?"}'
```

`/query` returns `answer`, `citations` (source, page, relevance), `route`, `sql_queries`, `generated_by`, `refused`, `refusal_reason`, `filters`, `latency_ms`, and `usage` (including prompt-cache reads). `/query/stream` sends server-sent events: `meta` (route and citations), then `token` deltas, then `done` with the full payload.

- **Auth:** set `RAG_API_KEYS=key1,key2` and clients send `x-api-key` or `Authorization: Bearer`. With no keys configured, auth is off, which is meant for local development only.
- **Rate limiting:** a token bucket per API key (or per client address without auth), configured with `RAG_RATE_LIMIT_PER_MINUTE` and `RAG_RATE_LIMIT_BURST`. Exceeding it returns `429` with `Retry-After`. Failed authentication attempts spend the caller's budget, so guessing keys is throttled.
- **Tracing:** `RAG_TRACING=otlp` exports spans through the standard `OTEL_EXPORTER_OTLP_*` variables, to Jaeger, Tempo, or Langfuse (see `.env.example`).
- **pgvector:** `docker compose --profile pgvector up`, then `RAG_VECTOR_STORE=pgvector` and `RAG_DATABASE_URL`.

## Evaluation

```bash
make eval                                             # the gate CI runs
make compare                                          # all retrievers + ablations -> evals/baselines/
uv run retail-rag eval --reranker cross-encoder       # try the reranker
uv run retail-rag eval --no-metadata-filters          # ablation
uv run retail-rag eval --judge --stem claude          # with ANTHROPIC_API_KEY: generation, SQL, LLM judge
```

| Metric | What it measures |
|---|---|
| Hit rate@k / Recall@k | Did the right evidence reach the model at all? |
| MRR / nDCG@k | Was the right evidence ranked near the top? |
| Refusal accuracy / false refusal rate | Does the system decline what the corpus cannot answer, without declining what it can? |
| Refusal layers | Which layer refused each unanswerable question: metadata filter, evidence gate, or model |
| Route accuracy | RAG / SQL / hybrid routing on the golden set, and on a held-out set never used for tuning |
| SQL execution accuracy | Does the agent's final query return the same rows as the gold query? (Claude mode) |
| Answer accuracy, citation validity, faithfulness | Generation quality; faithfulness is graded by an LLM judge with structured output (Claude mode) |

The golden set format is documented in [`evals/README.md`](evals/README.md). Baseline reports are committed in [`evals/baselines/`](evals/baselines/).

### Results

All runs use the same 230 questions, `k=4`, local mode (no generation), and metadata filters on unless stated. Full reports are in [`evals/baselines/`](evals/baselines/).

| Configuration | Hit@4 | Recall@4 | MRR | nDCG@4 | Paraphrase Hit@4 | Refusal accuracy | False refusals | Pass rate |
|---|---|---|---|---|---|---|---|---|
| TF-IDF (baseline) | 0.933 | 0.922 | 0.870 | 0.876 | 0.577 | 0.269 | 0.000 | 0.861 |
| BM25 | 0.950 | 0.939 | 0.892 | 0.896 | 0.692 | 0.346 | 0.006 | 0.883 |
| Dense (bge-small) | 0.956 | 0.944 | 0.853 | 0.868 | 0.923 | 0.500 | 0.000 | 0.904 |
| **Hybrid (BM25 + dense, RRF): default** | **0.978** | **0.964** | **0.900** | **0.908** | **0.885** | **0.538** | **0.000** | **0.926** |
| Hybrid, metadata filters off | 0.978 | 0.964 | 0.898 | 0.907 | 0.885 | 0.269 | 0.000 | 0.896 |
| Hybrid + cross-encoder rerank | 0.994 | 0.986 | 0.940 | 0.945 | 0.962 | 0.500 | 0.000 | 0.935 |

- **Routing:** 0.996 on the golden set and **0.947 on the held-out set**. The rules were written against the golden set, so the held-out figure is the honest one.
- **Latency on a GitHub Actions runner (2 vCPU):** hybrid retrieval takes 7.7 ms p50. Adding the reranker brings it to 458 ms p50, spent scoring 20 query–chunk pairs.

### Findings

1. **Metadata filters doubled refusal accuracy for free.** Hybrid retrieval with filters off refuses 0.269 of unanswerable questions; with filters on it refuses 0.538, with the same retrieval quality and zero false refusals. The filters catch what similarity cannot: "Coles FY24 revenue" embeds almost identically to "Coles FY25 revenue", but the corpus has no FY24 Coles document, so the filter leaves nothing to answer from.
2. **The reranker is the best retrieval lever, at a latency price.** It lifts MRR from 0.900 to 0.940 and paraphrase hit rate from 0.885 to 0.962. It fixes three of the four remaining retrieval misses, for example "never picks up their online order" versus "uncollected orders are cancelled". It costs about 450 ms of CPU per query, so it is off by default. It is worth enabling when Claude generation already dominates latency, or on a GPU.
3. **Reranking also moves the evidence gate.** The gate reads the dense similarity of the evidence actually returned, and the reranker returns different evidence. One off-topic question ("RBA cash rate") passes as a result, which is why refusal accuracy is 0.500 rather than 0.538. The threshold is calibrated per configuration, and each report's sweep table shows the trade-off.
4. **The gate's margin shrinks as the corpus grows.** With 7 documents, off-topic questions topped out at 0.49 and answerable ones started at 0.56. With 12 documents, the figures are 0.56 and 0.59. The gate was recalibrated from 0.52 to 0.575. This is why the evaluation report includes a threshold sweep, and why the gate is re-checked on every corpus change.
5. **In-domain unanswerable questions need the model.** "Harbourline's share price" or "the CEO's name" score 0.68–0.78, higher than many real answers. After the two cheap layers, 12 of 26 unanswerable questions remain for Claude's refusal contract, which is measured in Claude mode.
6. **Lexical retrieval breaks on paraphrases.** Paraphrase hit rate is 0.58 for TF-IDF, 0.69 for BM25, 0.92 for dense, and 0.96 with the reranker. Hybrid (0.885) trades a little paraphrase recall against dense-only retrieval for the best overall ranking.
7. **A metric bug was found and fixed.** nDCG credited every chunk matching a label, so two chunks with the same figure pushed it above 1. Each label is now credited once, with a regression test. The nDCG values in earlier versions of this README were inflated by this bug.

With 230 questions, one question is worth about 0.4 points of a metric (0.6 points over the 180 answerable document questions, 3.8 points over the 26 unanswerable ones). Differences smaller than that are noise.

## Design decisions

| Decision | Why | Trade-off |
|---|---|---|
| Hybrid BM25 + dense with RRF | RRF fuses ranks, so no score normalisation or tuning is needed across BM25 (unbounded) and cosine ([0, 1]) | Slightly below dense-only on paraphrases |
| Reranker off by default | About 60× the retrieval latency on CPU for +4 MRR points | Enable with `RAG_RERANKER=cross-encoder` when latency allows |
| Gate on dense similarity, not reranker scores | Cosine is calibrated and stable; cross-encoder logits are not | The gate still depends on which evidence is returned |
| Company filter strict, fiscal-year filter keeps undated documents | Policies are timeless: "FY25 out-of-stock rate vs the policy target" needs both the FY25 report and the undated policy | A wrong-year question about Harbourline still reaches the gate and the model |
| Rule-based router | Instant, free, deterministic, and explainable (each decision lists the rules that fired) | 0.947 held-out; an LLM classifier is the upgrade path |
| SQL tool safety in layers | Model-written SQL is untrusted: `mode=ro` connection, SQLite authorizer allow-list (only SELECT, reads, functions), one statement per call, time budget, row cap | Deliberately read-only; no write-back actions |
| Execution accuracy with tolerance | Extra columns, row order, and rounding within 0.5% shouldn't count as wrong | May accept a query that is right for the wrong reason |
| Prompt caching on stable prefixes | The SQL agent's ~1.1k-token schema prompt is cached, and the tool loop re-reads it every step | The RAG system prompt (~200 tokens) is below Claude Opus 5's 512-token minimum, so it isn't cached yet; `usage.cache_read_input_tokens` shows this directly |
| In-process rate limiter | No infrastructure; correct for one replica | With several replicas, move the state to Redis or the gateway |

## Engineering

| Concern | Implementation |
|---|---|
| Dependencies | `uv` + committed `uv.lock`; optional extras `api`, `llm`, `pdf`, `embeddings`, `pgvector`, `otel`, `dev` |
| Config | `pydantic-settings` with validation (`src/retail_rag/config.py`); secrets held as `SecretStr` |
| Quality | `ruff` (lint + format + bandit rules), `mypy --strict`, `pre-commit` |
| Tests | 186 pytest tests: unit, API, CLI, evaluation, retrievers, SQL safety, routing, streaming, tracing, auth. Claude is replaced by a scripted fake, so the suite is offline and deterministic. CI also runs a real bge-small integration test and pgvector tests against a Postgres service container. Coverage gate 85% (currently 92%) |
| CI | lint → tests (py3.11/3.12, with Postgres + pgvector) → eval gate (retrieval, refusal, routing, held-out routing) plus an informational reranker run → Docker build + smoke test; a manual live-Claude eval job |
| Container | multi-stage build, non-root user, index and SQL database baked in, `HEALTHCHECK` on `/ready`, JSON logs |
| Security | API-key auth (digests compared in constant time), rate limiting, XML-escaped evidence against prompt injection, read-only SQL with an authorizer, no secrets in logs (key fingerprints only) |
| Observability | OpenTelemetry spans (`rag.query` → `rag.route` / `rag.retrieve` / `gen_ai.generate` / `rag.sql_agent` → `rag.sql.execute`) with GenAI token and cache attributes; structured logs with `request_id`, route, refusal reason, latency, and usage |

```bash
make check   # lint + typecheck + tests + eval gate: everything CI runs
```

## Repository layout

```text
src/retail_rag/
  config.py          typed settings from env / .env
  ingest.py          front matter + sidecar metadata, page-aware PDF parsing
  chunking.py        deterministic chunking
  filters.py         self-query metadata filters (company, fiscal year)
  retrieval/         TF-IDF, BM25, dense, pgvector, hybrid (RRF), cross-encoder rerank
  router.py          rule-based RAG / SQL / hybrid router
  sql/               synthetic database, read-only SQL tool, Claude text-to-SQL agent
  generation.py      Claude generation: XML context, refusal contract, caching, streaming
  pipeline.py        orchestration and layered refusal
  api.py, security.py  FastAPI service, auth, rate limiting, SSE streaming
  tracing.py         OpenTelemetry spans (no-op when off)
  cli.py             ingest / query / serve / eval
  evaluation/        golden-set schema, metrics, runner, reports, LLM judge, held-out routing
evals/               golden set, held-out routing set, committed baseline reports
data/documents/      public snapshot, synthetic policies, synthetic annual report PDF
scripts/             reproducible generator for the synthetic annual report PDF
tests/               pytest suite
```

## Data and claim boundaries

- The Coles snapshot links back to the official source page. Refresh it before a public demo.
- All Harbourline documents, the annual report, and the operational database are synthetic. Their volumes are illustrative and are not reconciled with each other. Transactional data must stay synthetic unless there is explicit permission to use real data.
- Do not describe this repository as a Coles deployment or customer engagement.
- Do not commit API keys, private documents, personal data, or former-employer data.

## Roadmap

1. ~~Evaluation harness with golden set and CI quality gate~~ ✅
2. ~~Engineering baseline: uv, ruff, mypy, Docker, CI, structured logging~~ ✅
3. ~~Dense embeddings and hybrid BM25/vector retrieval with RRF~~ ✅
4. ~~Cross-encoder reranker, and pgvector (HNSW) behind the same retriever interface~~ ✅
5. ~~Page-aware PDF parsing with company, fiscal-year, and page metadata filters~~ ✅
6. ~~Synthetic orders and inventory, a read-only SQL tool, and a RAG / SQL / hybrid router~~ ✅
7. ~~Tracing (OpenTelemetry/Langfuse), prompt caching, streaming responses, auth and rate limiting~~ ✅
8. Run the Claude-mode evaluation in CI (needs the `ANTHROPIC_API_KEY` repository secret) and publish answer accuracy, faithfulness, and SQL execution accuracy
9. LLM-based router as a fallback when the rules are unsure; Redis-backed rate limiting for multiple replicas
10. Layout-aware table extraction for real, multi-column annual reports
