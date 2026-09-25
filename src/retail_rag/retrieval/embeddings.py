"""Embedding providers.

``Embedder`` is the seam for swapping models or vendors (a hosted API such as
Voyage, or a different local model) without touching retrieval code. The default
implementation runs ``BAAI/bge-small-en-v1.5`` locally through fastembed (ONNX
Runtime, no PyTorch), so no data leaves the machine and no API key is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Return one L2-normalised row per text."""
        ...

    def embed_query(self, text: str) -> Vectors:
        """Return a single L2-normalised vector (1-D)."""
        ...


def normalise(matrix: Vectors) -> Vectors:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed.

    ``model_path`` loads a pre-downloaded model directory, for air-gapped
    deployments or images that must not download at runtime.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        model_path: Path | None = None,
        cache_dir: Path | None = None,
    ):
        try:
            from fastembed import TextEmbedding  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "Dense retrieval needs the 'embeddings' extra: uv sync --extra embeddings"
            ) from exc

        self.model_name = model_name
        self._model = TextEmbedding(
            model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
            specific_model_path=str(model_path) if model_path else None,
        )

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        vectors = np.asarray(list(self._model.passage_embed(list(texts))), dtype=np.float32)
        return normalise(vectors)

    def embed_query(self, text: str) -> Vectors:
        vector = np.asarray(next(iter(self._model.query_embed([text]))), dtype=np.float32)
        return normalise(vector)
