"""Document ingestion: parse, attach metadata, chunk.

Metadata comes from two places and is merged (sidecar wins):

* Markdown/text **front matter**, a leading block of ``key: value`` lines between
  ``---`` fences.
* A **sidecar** ``<file>.meta.json`` next to any document, which is how PDFs
  (which cannot carry front matter) get their company and fiscal year.

``company`` and ``fiscal_year`` are the filterable keys (see ``filters.py``) and
are always stored as lists, because one report can cover several years: an FY25
annual report also states FY24 comparatives.

PDFs are parsed page by page, so every chunk records the ``page`` it came from
and citations can point at it. Lines that repeat on most pages (running headers
and footers such as "Annual Report 2025 | Page 3") are removed before chunking,
otherwise they end up in every chunk and dilute retrieval.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .chunking import chunk_text
from .config import Settings, get_settings
from .models import DocumentChunk
from .retrieval import Retriever, create_retriever, embeddings_cache_path, save_chunks

if TYPE_CHECKING:
    from .retrieval.embeddings import Embedder

SUPPORTED_TEXT_EXTENSIONS = {".md", ".txt", ".markdown"}
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | {".pdf"}
LIST_KEYS = ("company", "fiscal_year")
FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.DOTALL)


@dataclass(frozen=True)
class Page:
    number: int | None
    """1-based page number for PDFs; ``None`` for formats without pages."""
    text: str


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if "," in raw:
        return [_parse_value(part) for part in raw.split(",") if part.strip()]
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw.strip("\"'")


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split ``---``-fenced ``key: value`` front matter from the document body."""
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    metadata: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            metadata[key.strip()] = _parse_value(value)
    return metadata, text[match.end() :]


def normalise_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalised = dict(metadata)
    for key in LIST_KEYS:
        if key in normalised and not isinstance(normalised[key], list):
            normalised[key] = [normalised[key]]
    return normalised


def _normalise_line(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _edge_lines(page: str, depth: int) -> list[str]:
    lines = [line for line in page.splitlines() if line.strip()]
    return lines[:depth] + lines[-depth:]


def strip_repeated_lines(pages: list[str], *, min_share: float = 0.6, depth: int = 1) -> list[str]:
    """Remove running headers/footers: the top or bottom ``depth`` lines of a
    page that recur (digits ignored, so "Page 3 of 9" matches) on most pages.

    Only the page edges are considered, so a body line that happens to repeat
    (a table row label, a recurring sentence) is never removed.
    """
    if len(pages) < 3:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_normalise_line(line) for line in _edge_lines(page, depth)})
    repeated = {line for line, count in counts.items() if count / len(pages) >= min_share}
    cleaned = []
    for page in pages:
        lines = [line for line in page.splitlines() if line.strip()]
        keep = [
            line
            for index, line in enumerate(lines)
            if not (
                (index < depth or index >= len(lines) - depth) and _normalise_line(line) in repeated
            )
        ]
        cleaned.append("\n".join(keep).strip())
    return cleaned


def _read_pdf(path: Path) -> list[Page]:
    try:
        from pypdf import PdfReader  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise RuntimeError(
            "PDF support is optional. Install it with: python -m pip install -e '.[pdf]'"
        ) from exc
    reader = PdfReader(str(path))
    texts = [page.extract_text() or "" for page in reader.pages]
    return [
        Page(number, text)
        for number, text in enumerate(strip_repeated_lines(texts), start=1)
        if text.strip()
    ]


def read_document(path: Path) -> tuple[list[Page], dict[str, Any]]:
    """Return the document's pages and its metadata (front matter + sidecar)."""
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        pages = [Page(None, body)]
    elif suffix == ".pdf":
        metadata, pages = {}, _read_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    sidecar = path.with_name(f"{path.name}.meta.json")
    if sidecar.is_file():
        metadata.update(json.loads(sidecar.read_text(encoding="utf-8")))
    return pages, normalise_metadata(metadata)


def load_chunks(
    documents_dir: Path, *, chunk_size: int = 900, overlap: int = 120
) -> list[DocumentChunk]:
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")

    chunks: list[DocumentChunk] = []
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("README"):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        pages, doc_metadata = read_document(path)
        relative_source = path.relative_to(documents_dir).as_posix()
        index = 0
        for page in pages:
            for piece in chunk_text(page.text, chunk_size=chunk_size, overlap=overlap):
                # Chunks never span pages, so every chunk has exactly one page to cite.
                location = "" if page.number is None else f"p{page.number}:"
                raw_id = f"{relative_source}:{location}{index}:{piece}"
                metadata = {
                    **doc_metadata,
                    "chunk_index": index,
                    "file_type": path.suffix.lower(),
                }
                if page.number is not None:
                    metadata["page"] = page.number
                chunks.append(
                    DocumentChunk(
                        chunk_id=hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12],
                        source=relative_source,
                        text=piece,
                        metadata=metadata,
                    )
                )
                index += 1
    return chunks


def build_index(
    documents_dir: Path,
    index_path: Path,
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
) -> Retriever:
    """Chunk the documents, persist them, and build (and cache) the configured retriever."""
    settings = settings or get_settings()
    chunks = load_chunks(
        documents_dir, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    if not chunks:
        raise RuntimeError(f"No supported documents found in {documents_dir}")
    save_chunks(chunks, index_path)
    return create_retriever(
        settings.retriever,
        chunks,
        settings,
        cache_path=embeddings_cache_path(index_path),
        embedder=embedder,
    )
