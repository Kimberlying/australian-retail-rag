# Australian Retail RAG

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-lightgrey.svg)](README.zh-CN.md)
[![CI](https://github.com/kimberlying/australian-retail-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/kimberlying/australian-retail-rag/actions/workflows/ci.yml)

A retrieval-augmented generation (RAG) system for Australian retail, built around evaluation. It uses public Australian company material alongside clearly labelled synthetic retail data.

It is an independent portfolio project. It is **not a Coles system, customer project, or endorsement**. "Harbourline Retail" in the synthetic documents is a fictional company.

## Highlights

- **Evaluation harness:** a 55-question golden set with retrieval metrics (Hit@k, Recall@k, MRR, nDCG, Precision@k) and generation metrics (answer accuracy, refusal accuracy, citation validity, and LLM-as-judge faithfulness). CI fails when retrieval or refusal quality drops.
- **Hybrid retrieval:** BM25 and dense embeddings (`bge-small-en-v1.5`, run locally through ONNX) fused with Reciprocal Rank Fusion. It was chosen over TF-IDF, BM25, and dense-only retrieval by measurement; see the [comparison](#retriever-comparison) below.
- **Calibrated evidence gate:** off-topic questions are refused before any LLM call, based on a dense-similarity threshold chosen from a threshold sweep in the evaluation report.
- **Grounded generation with Claude:** evidence is passed to the model as XML, which helps resist prompt injection. Answers include citations, and refusals are machine-detectable. Server-side refusal fallback is enabled, and the system degrades to a local retrieval preview when there is no API key or the API call fails.
- **Production engineering:** typed settings (pydantic-settings), structured JSON logs with request ids, a FastAPI service that loads the index once at startup with separate liveness and readiness probes, a multi-stage non-root Docker image, a `uv` lockfile, ruff, strict mypy, pytest with coverage, pre-commit, and GitHub Actions.

## Architecture

```text
 data/documents/*.md|pdf ──► ingest ──► chunk ──► chunks.json + embeddings.npz (bge-small, cached)
                                                                     │
 question ──┬─► BM25 (exact terms, numbers) ──┐
            └─► dense cosine (paraphrases) ───┴─► RRF fusion ──► top-k
                                                                     │
                            evidence gate: best dense similarity < 0.52 ? ──┬── yes ──► refuse (no LLM call)
                                                                            │
                                                                            └── no ───► Claude (XML-wrapped context)
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

The first `ingest` downloads the embedding model (about 70 MB) and caches the chunk embeddings next to the index. For a fully offline run, set `RAG_RETRIEVER=bm25`; to use a pre-downloaded model directory, set `RAG_EMBEDDING_MODEL_PATH`.

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

### Retriever comparison

All runs use the same 55 questions, `k=4`, local mode, and each retriever's calibrated gate. The full reports are in [`evals/baselines/`](evals/baselines/), and `make compare` regenerates them.

| Retriever | Hit@4 | Recall@4 | MRR | nDCG@4 | Paraphrase Hit@4 | Refusal accuracy | False refusals | Pass rate | p50 latency |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF (baseline) | 0.955 | 0.932 | 0.917 | 0.954 | 0.778 | 0.000 | 0 | 0.764 | 0.1 ms |
| BM25 | 0.977 | 0.955 | 0.932 | 0.967 | 0.889 | 0.182 | 0 | 0.818 | 0.03 ms |
| Dense (bge-small) | 0.955 | 0.955 | 0.884 | 0.922 | 0.889 | 0.273 | 0 | 0.818 | 2.9 ms |
| **Hybrid (BM25 + dense, RRF)** | **0.977** | **0.955** | **0.932** | **0.969** | **0.889** | **0.273** | **0** | **0.836** | 2.5 ms |

Findings:

1. **Lexical retrieval breaks on paraphrases.** With TF-IDF, "freezer" does not match "frozen". Dense retrieval handles the paraphrase but ranks exact facts lower (MRR 0.884). Hybrid keeps both strengths, and it is the default for that reason.
2. **BM25 alone is a stronger baseline than TF-IDF.** Stopword removal plus length normalisation lifted paraphrase hit rate from 0.78 to 0.89 without any model, which confirms that TF-IDF was a weak baseline rather than lexical search being inherently weak.
3. **A similarity threshold separates off-topic questions but not hard negatives.** The [threshold sweep](evals/baselines/hybrid_local.md#evidence-gate-threshold-sweep) shows:
   - The clearly unrelated questions (capital of France, a poem request, and the prompt injection) score ≤ 0.49, and every answerable question scores ≥ 0.56. The gate sits at 0.52 and refuses those 3 without an LLM call and with zero false refusals.
   - Near-domain questions pass the gate, because they are retail or finance questions the corpus happens not to cover: "RBA cash rate" scores 0.63 and "Harbourline stores in WA" scores 0.65.
   - "Coles **FY24** revenue" scores **0.78**, higher than most real answers: it is on topic, and only the year is wrong. TF-IDF could not even separate the off-topic questions, because "What is the capital of France?" scored 0.335 through stopwords alone.
   - Refusal is therefore layered: the cheap retrieval gate handles off-topic questions, and the generator's refusal contract, measured by `refusal_accuracy` in Claude mode, handles near-domain, wrong-entity, and wrong-period questions.
4. **The one remaining answerable miss** is a paraphrase: "someone stealing" vs "shoplifter". A cross-encoder reranker is the next lever.

With 55 questions, one question is worth about 2 points of any metric. Treat differences below that as noise, and grow the golden set before tuning fusion weights.

## Engineering

| Concern | Implementation |
|---|---|
| Dependencies | `uv` + committed `uv.lock`; optional extras `api`, `llm`, `pdf`, `embeddings`, `dev` |
| Config | `pydantic-settings` with validation (`src/retail_rag/config.py`); secrets are held as `SecretStr` |
| Quality | `ruff` (lint + format + bandit rules), `mypy --strict`, `pre-commit` |
| Tests | 91 pytest tests covering unit, API, CLI, eval, and retrievers. Claude and the embedding model are faked, so the suite is offline and deterministic. One integration test runs the real bge-small model in CI. The CI coverage gate is 85% |
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
  retrieval/         TF-IDF, BM25, dense, hybrid (RRF) retrievers; embeddings; factory
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
3. ~~Dense embeddings and hybrid BM25/vector retrieval with RRF, compared against the TF-IDF baseline~~ ✅
4. Cross-encoder reranker, and pgvector (HNSW) behind the same retriever interface
5. Layout-aware PDF parsing of real annual reports, with company, year, and page metadata filters
6. Synthetic orders and inventory in Postgres, a read-only SQL tool, and a RAG / SQL / hybrid router
7. Tracing (Langfuse/OpenTelemetry), prompt caching, streaming responses, auth and rate limiting
