from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import Settings, get_settings
from .generation import REFUSAL_TEXT, generate_with_claude, local_preview
from .models import Answer, GeneratedBy
from .retrieval import Retriever, load_retriever

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class RAGPipeline:
    def __init__(self, retriever: Retriever, settings: Settings | None = None):
        self.retriever = retriever
        self.settings = settings or get_settings()

    @property
    def min_score(self) -> float:
        """Evidence-gate threshold: explicit setting, else the retriever's calibrated default."""
        if self.settings.min_score is not None:
            return self.settings.min_score
        return self.retriever.default_min_score

    def ask(self, question: str, *, top_k: int | None = None) -> Answer:
        k = top_k or self.settings.top_k
        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        retrievals = self.retriever.search(question, top_k=k)
        # Answer-level evidence gate: refuse when even the best chunk is not similar
        # enough. Individual chunks are not filtered, so a hybrid result that BM25
        # found by exact term match survives even with modest dense similarity.
        relevances = [item.relevance for item in retrievals if item.relevance is not None]
        evidence_score = max(relevances) if relevances else None
        if evidence_score is not None and evidence_score < self.min_score:
            retrievals = []
        latency = {"retrieval": _elapsed_ms(retrieval_start)}

        generation_start = time.perf_counter()
        generated_by: GeneratedBy = "local"
        refused = not retrievals
        usage: dict[str, int] = {}
        if not retrievals:
            # Nothing cleared the relevance threshold: refuse without spending an LLM call.
            answer_text = REFUSAL_TEXT
        else:
            try:
                generation = generate_with_claude(question, retrievals, self.settings)
            except Exception:  # API/network errors must not take down retrieval.
                logger.exception("Claude generation failed; falling back to local preview")
                generation = None
                generated_by = "local_fallback"
            if generation is not None:
                answer_text, refused, usage = generation.text, generation.refused, generation.usage
                generated_by = "claude"
            else:
                answer_text = local_preview(question, retrievals)
        latency["generation"] = _elapsed_ms(generation_start)
        latency["total"] = _elapsed_ms(total_start)

        citations = [
            {
                "source": item.chunk.source,
                "chunk_id": item.chunk.chunk_id,
                "score": round(item.score, 4),
                "relevance": None if item.relevance is None else round(item.relevance, 4),
            }
            for item in retrievals
        ]
        logger.info(
            "query answered",
            extra={
                "fields": {
                    "generated_by": generated_by,
                    "refused": refused,
                    "n_retrieved": len(retrievals),
                    "retriever": self.retriever.name,
                    "evidence_score": evidence_score,
                    "latency_ms": latency["total"],
                    **usage,
                }
            },
        )
        return Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrievals=retrievals,
            generated_by=generated_by,
            refused=refused,
            evidence_score=evidence_score,
            latency_ms=latency,
            usage=usage,
        )

    @classmethod
    def from_index(cls, index_path: Path, settings: Settings | None = None) -> RAGPipeline:
        settings = settings or get_settings()
        return cls(load_retriever(index_path, settings), settings)
