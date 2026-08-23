# Australian Retail RAG

An interview-ready RAG starter project using public Australian company material and clearly labelled synthetic retail data.

The first demo uses a short public-data snapshot from Coles and a synthetic retail operations policy. It is an independent portfolio simulation — **not a Coles system, customer project, or endorsement**.

## Why this project exists

The project connects two real engineering backgrounds without rewriting history:

- Data Engineering: ingestion, chunking, metadata, data quality, and structured-data boundaries.
- AI Engineering: retrieval, grounded answers, citations, evaluation, and an optional Claude generation layer.

The next iteration can add a read-only SQL tool for synthetic orders and inventory, then route questions to RAG, SQL, or both.

## Architecture

```text
Public reports / policies
          |
          v
   document ingestion
          |
          v
 chunking + TF-IDF vector index  --->  top-k evidence
          |                                  |
          +----------------------------> Claude API (optional)
                                             |
                                             v
                                  answer + source citations
```

The local TF-IDF index is deliberately dependency-light and easy to explain in an interview. It is a working baseline, not a claim that TF-IDF is the final production embedding strategy. The upgrade path is `embedding provider -> pgvector/Qdrant -> evaluation harness`.

## Run it locally

```bash
cd australian-retail-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

retail-rag ingest
retail-rag query "What was Coles' normalised eCommerce sales growth in FY25?"
```

Without `ANTHROPIC_API_KEY`, the query command returns the retrieved evidence as a local preview. To generate a grounded Claude answer:

```bash
cp .env.example .env
# Add the key locally; never commit .env.
export ANTHROPIC_API_KEY="your-key"
retail-rag query "What was Coles' normalised eCommerce sales growth in FY25?"
```

Optional API mode:

```bash
python -m pip install -e ".[api]"
retail-rag serve
curl -X POST http://127.0.0.1:8000/query \\
  -H 'content-type: application/json' \\
  -d '{"question":"What was Coles normalised eCommerce sales growth in FY25?"}'
```

Optional PDF ingestion:

```bash
python -m pip install -e ".[pdf]"
```

Place permitted public `.pdf`, `.md`, or `.txt` files under `data/documents/`, then rerun `retail-rag ingest`.

## Repository layout

```text
src/retail_rag/
  chunking.py       deterministic text chunking
  retrieval.py      local TF-IDF vector retrieval and persistence
  ingest.py         markdown/text/PDF ingestion
  generation.py     optional Claude generation with citations
  pipeline.py       RAG orchestration
  api.py            optional FastAPI endpoint
  cli.py            ingest/query/serve commands
data/documents/     public snapshot + synthetic policy
tests/              dependency-light regression tests
```

## Data and claim boundaries

- The Coles snapshot links back to the official source page and should be refreshed before a public demo.
- Transactional data should be synthetic unless there is explicit permission to use real data.
- Do not describe this repository as a Coles deployment or customer engagement.
- Do not commit API keys, private documents, personal data, or former-employer data.

## Interview talking points

> I built an independent public-data RAG prototype for Australian retail. The pipeline ingests source documents, chunks them with metadata, retrieves evidence, and returns a grounded answer with citations. I kept the local baseline dependency-light, then designed an upgrade path to real embeddings, a vector database, read-only SQL tools, evaluation, and production observability.

## Roadmap

1. Add synthetic orders/inventory in SQLite and a read-only SQL tool.
2. Add a query router for RAG vs SQL vs hybrid questions.
3. Replace TF-IDF with a real embedding provider and pgvector/Qdrant.
4. Add a 30–50 question golden set for retrieval and citation evaluation.
5. Add Docker, structured logs, latency/cost tracking, and a small demo UI.
