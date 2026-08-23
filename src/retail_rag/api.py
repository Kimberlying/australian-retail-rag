from __future__ import annotations

from pathlib import Path

from .config import index_path
from .pipeline import RAGPipeline

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without optional API deps
    raise RuntimeError("Install API dependencies with: python -m pip install -e '.[api]'") from exc


app = FastAPI(title="Australian Retail RAG", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=4, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "index": str(index_path())}


@app.post("/query")
def query(request: QueryRequest) -> dict[str, object]:
    path = Path(index_path())
    if not path.exists():
        raise HTTPException(status_code=503, detail="Build the index first with `retail-rag ingest`.")
    result = RAGPipeline.from_index(path).ask(request.question, top_k=request.top_k)
    return {
        "question": result.question,
        "answer": result.answer,
        "citations": result.citations,
    }
