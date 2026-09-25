from __future__ import annotations

import hashlib
from pathlib import Path

from .chunking import chunk_text
from .models import DocumentChunk
from .retrieval import TfidfRetriever

SUPPORTED_TEXT_EXTENSIONS = {".md", ".txt", ".markdown"}


def _read_file(path: Path) -> str:
    if path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "PDF support is optional. Install it with: python -m pip install -e '.[pdf]'"
            ) from exc
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_chunks(
    documents_dir: Path, *, chunk_size: int = 900, overlap: int = 120
) -> list[DocumentChunk]:
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")

    chunks: list[DocumentChunk] = []
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("README"):
            continue
        if path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS | {".pdf"}:
            continue
        text = _read_file(path)
        relative_source = path.relative_to(documents_dir).as_posix()
        for index, piece in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            raw_id = f"{relative_source}:{index}:{piece}"
            chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=relative_source,
                    text=piece,
                    metadata={"chunk_index": index, "file_type": path.suffix.lower()},
                )
            )
    return chunks


def build_index(
    documents_dir: Path, index_path: Path, *, chunk_size: int = 900, overlap: int = 120
) -> TfidfRetriever:
    chunks = load_chunks(documents_dir, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise RuntimeError(f"No supported documents found in {documents_dir}")
    retriever = TfidfRetriever(chunks)
    retriever.save(index_path)
    return retriever
