# syntax=docker/dockerfile:1.7

# ---------- builder: resolve locked dependencies into a virtualenv ----------
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir "uv==0.8.*"
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first so this layer is cached until the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra api --extra llm --extra pdf --extra embeddings

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra api --extra llm --extra pdf --extra embeddings

# ---------- runtime: minimal image, non-root user ----------
FROM python:3.12-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app
# /app itself is root-owned; pre-create only the paths the app user writes to
# (index + embeddings at build time, model cache).
RUN install -d -o app -g app /app/data /app/data/index /app/.cache
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app data/documents ./data/documents

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    RAG_DOCS_DIR=/app/data/documents \
    RAG_INDEX_PATH=/app/data/index/index.json \
    RAG_EMBEDDING_CACHE_DIR=/app/.cache/fastembed \
    LOG_FORMAT=json

USER app

# Bake the index, chunk embeddings, and embedding model into the image so
# containers start ready to serve and never download anything at runtime.
RUN retail-rag ingest
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status == 200 else 1)"

CMD ["retail-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
