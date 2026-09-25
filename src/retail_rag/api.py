from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from . import __version__
from .config import get_settings
from .logging_config import configure_logging
from .pipeline import RAGPipeline

try:
    from fastapi import FastAPI, HTTPException, Request, Response
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
    score: float
    relevance: float | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    generated_by: str
    refused: bool
    evidence_score: float | None
    latency_ms: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the index once at startup instead of on every request."""
    configure_logging()
    settings = get_settings()
    app.state.pipeline = None
    if settings.index_path.exists():
        app.state.pipeline = RAGPipeline.from_index(settings.index_path, settings)
        logger.info(
            "index loaded",
            extra={
                "fields": {
                    "chunks": len(app.state.pipeline.retriever.chunks),
                    "retriever": app.state.pipeline.retriever.name,
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
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        },
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready(request: Request) -> dict[str, Any]:
    """Readiness: the index is loaded and queries can be served."""
    pipeline: RAGPipeline | None = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return {
        "status": "ready",
        "chunks": len(pipeline.retriever.chunks),
        "retriever": pipeline.retriever.name,
        "llm_enabled": pipeline.settings.llm_enabled,
    }


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    pipeline: RAGPipeline | None = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=503, detail="Build the index first with `retail-rag ingest`."
        )
    result = pipeline.ask(body.question, top_k=body.top_k)
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        citations=[Citation(**citation) for citation in result.citations],
        generated_by=result.generated_by,
        refused=result.refused,
        evidence_score=result.evidence_score,
        latency_ms=result.latency_ms,
    )
