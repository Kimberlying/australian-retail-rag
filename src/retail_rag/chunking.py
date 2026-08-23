from __future__ import annotations

import re


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Split text into deterministic, character-bounded chunks.

    This is intentionally simple for the first commit. A production version can
    add layout-aware PDF parsing and token-aware chunking while preserving the
    same interface.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    cleaned = _normalise(text)
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            boundary = cleaned.rfind("\n\n", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
            else:
                boundary = cleaned.rfind(" ", start, end)
                if boundary > start + chunk_size // 2:
                    end = boundary

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks
