from __future__ import annotations

from pathlib import Path

from .generation import generate_with_claude, local_preview
from .models import Answer
from .retrieval import TfidfRetriever


class RAGPipeline:
    def __init__(self, retriever: TfidfRetriever):
        self.retriever = retriever

    def ask(self, question: str, *, top_k: int = 4) -> Answer:
        retrievals = self.retriever.search(question, top_k=top_k)
        try:
            answer = generate_with_claude(question, retrievals)
        except Exception as exc:  # API/network errors should not destroy local retrieval.
            answer = f"Claude generation failed ({type(exc).__name__}); showing local evidence instead.\n\n"
            answer += local_preview(question, retrievals)
        answer = answer or local_preview(question, retrievals)
        citations = [
            {
                "source": item.chunk.source,
                "chunk_id": item.chunk.chunk_id,
                "score": round(item.score, 4),
            }
            for item in retrievals
        ]
        return Answer(
            question=question,
            answer=answer,
            citations=citations,
            retrievals=retrievals,
        )

    @classmethod
    def from_index(cls, index_path: Path) -> "RAGPipeline":
        return cls(TfidfRetriever.load(index_path))
