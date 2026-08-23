from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def docs_dir() -> Path:
    configured = os.getenv("RAG_DOCS_DIR", "data/documents")
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def index_path() -> Path:
    configured = os.getenv("RAG_INDEX_PATH", "data/index/index.json")
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def anthropic_model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
