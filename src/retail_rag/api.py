from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from . import __version__
from .config import get_settings
from .logging_config import configure_logging
from .pipeline import RAGPipeline, answer_payload
from .security import APIKeyAuth, RateLimiter, key_fingerprint
from .tracing import configure_tracing, tracing_mode

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without optional API deps
    raise RuntimeError("Install API dependencies with: python -m pip install -e '.[api]'") from exc

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Citation(BaseModel):
    source: str
    chunk_id: str
    page: int | None = None
    score: float
    relevance: float | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    route: str
    sql_queries: list[str]
    generated_by: str
    refused: bool
    refusal_reason: str | None
    filters: dict[str, list[Any]]
    evidence_score: float | None
    latency_ms: dict[str, float]
    usage: dict[str, int]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the index once at startup instead of on every request."""
    configure_logging()
    settings = get_settings()
    configure_tracing(settings.tracing)
    app.state.auth = APIKeyAuth(settings.api_key_list)
    app.state.limiter = RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)
    if not app.state.auth.enabled:
        logger.warning("RAG_API_KEYS is not set: /query is unauthenticated (local dev only)")
    app.state.pipeline = None
    if settings.index_path.exists():
        app.state.pipeline = RAGPipeline.from_index(settings.index_path, settings)
        logger.info(
            "index loaded",
            extra={
                "fields": {
                    "chunks": len(app.state.pipeline.retriever.chunks),
                    "retriever": app.state.pipeline.retriever.name,
                    "sql_enabled": app.state.pipeline.sql_tool is not None,
                }
            },
        )
    else:
        logger.warning("index not found; /query will return 503 until it is built")
    yield


app = FastAPI(title="Australian Retail RAG", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    logger.info(
        "http request",
        extra={
            "fields": {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "client": getattr(request.state, "client_id", None),
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        },
    )
    return response


def _presented_key(x_api_key: str | None, authorization: str | None) -> str | None:
    if x_api_key:
        return x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def protect(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """Authenticate (when keys are configured), then apply the per-client rate limit."""
    auth: APIKeyAuth = request.app.state.auth
    limiter: RateLimiter = request.app.state.limiter
    key = _presented_key(x_api_key, authorization)
    address = f"ip:{request.client.host if request.client else 'unknown'}"
    if not auth.verify(key):
        # Failed attempts spend the caller's address budget, so guessing keys is throttled.
        request.state.client_id = address
        _enforce(limiter, address)
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Limit per key when auth is on; per client address otherwise.
    client_id = f"key:{key_fingerprint(key)}" if auth.enabled and key else address
    request.state.client_id = client_id
    _enforce(limiter, client_id)
    return client_id


def _enforce(limiter: RateLimiter, client_id: str) -> None:
    wait = limiter.acquire(client_id)
    if wait > 0:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": RateLimiter.retry_after_header(wait)},
        )


def _pipeline(request: Request) -> RAGPipeline:
    pipeline: RAGPipeline | None = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=503, detail="Build the index first with `retail-rag ingest`."
        )
    return pipeline


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready(request: Request) -> dict[str, Any]:
    """Readiness: the index is loaded and queries can be served."""
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return {
        "status": "ready",
        "chunks": len(pipeline.retriever.chunks),
        "retriever": pipeline.retriever.name,
        "vector_store": pipeline.settings.vector_store,
        "sql_enabled": pipeline.sql_tool is not None,
        "llm_enabled": pipeline.settings.llm_enabled,
        "auth_enabled": request.app.state.auth.enabled,
        "tracing": tracing_mode(),
    }


@app.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest, request: Request, _client: Annotated[str, Depends(protect)]
) -> QueryResponse:
    result = _pipeline(request).ask(body.question, top_k=body.top_k)
    return QueryResponse.model_validate(answer_payload(result))


def _sse(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        name = event.pop("event")
        yield f"event: {name}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/query/stream")
def query_stream(
    body: QueryRequest, request: Request, _client: Annotated[str, Depends(protect)]
) -> StreamingResponse:
    """Server-sent events: ``meta`` (route + citations), ``token`` deltas, then ``done``."""
    pipeline = _pipeline(request)
    return StreamingResponse(
        _sse(pipeline.stream(body.question, top_k=body.top_k)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
