"""Typed, validated application settings.

All configuration comes from environment variables (or a local ``.env`` file)
so the same image runs unchanged in dev, CI, and production.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

RetrieverKind = Literal["tfidf", "bm25", "dense", "hybrid"]
RerankerKind = Literal["none", "cross-encoder"]
VectorStoreKind = Literal["memory", "pgvector"]
TracingMode = Literal["off", "console", "otlp"]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- data & index
    docs_dir: Path = Field(default=Path("data/documents"), alias="RAG_DOCS_DIR")
    index_path: Path = Field(default=Path("data/index/index.json"), alias="RAG_INDEX_PATH")
    chunk_size: int = Field(default=900, gt=0, alias="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, ge=0, alias="RAG_CHUNK_OVERLAP")

    # --- retrieval
    retriever: RetrieverKind = Field(default="hybrid", alias="RAG_RETRIEVER")
    top_k: int = Field(default=4, ge=1, le=20, alias="RAG_TOP_K")
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        alias="RAG_MIN_SCORE",
        description="Evidence-gate threshold; unset uses the retriever's calibrated default.",
    )
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="RAG_EMBEDDING_MODEL")
    embedding_model_path: Path | None = Field(default=None, alias="RAG_EMBEDDING_MODEL_PATH")
    embedding_cache_dir: Path | None = Field(default=None, alias="RAG_EMBEDDING_CACHE_DIR")
    vector_store: VectorStoreKind = Field(
        default="memory",
        alias="RAG_VECTOR_STORE",
        description="Where dense vectors live: in-process numpy, or Postgres + pgvector (HNSW).",
    )
    database_url: SecretStr | None = Field(default=None, alias="RAG_DATABASE_URL")
    reranker: RerankerKind = Field(default="none", alias="RAG_RERANKER")
    reranker_model: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2", alias="RAG_RERANKER_MODEL")
    reranker_model_path: Path | None = Field(default=None, alias="RAG_RERANKER_MODEL_PATH")
    rerank_candidates: int = Field(default=20, ge=1, le=100, alias="RAG_RERANK_CANDIDATES")
    metadata_filters: bool = Field(
        default=True,
        alias="RAG_METADATA_FILTERS",
        description="Restrict retrieval to the company / fiscal year a question names.",
    )

    # --- structured data (SQL route)
    sql_db_path: Path = Field(default=Path("data/index/retail.db"), alias="RAG_SQL_DB_PATH")
    router_enabled: bool = Field(default=True, alias="RAG_ROUTER_ENABLED")

    # --- generation
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-opus-5", alias="ANTHROPIC_MODEL")
    judge_model: str = Field(default="claude-sonnet-5", alias="RAG_JUDGE_MODEL")
    max_output_tokens: int = Field(default=2048, gt=0, alias="RAG_MAX_OUTPUT_TOKENS")
    llm_timeout_seconds: float = Field(default=60.0, gt=0, alias="RAG_LLM_TIMEOUT_SECONDS")

    # --- API protection
    api_keys: SecretStr | None = Field(
        default=None,
        alias="RAG_API_KEYS",
        description="Comma-separated API keys for /query; unset disables auth (local dev only).",
    )
    rate_limit_per_minute: int = Field(default=60, ge=0, alias="RAG_RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(default=10, ge=1, alias="RAG_RATE_LIMIT_BURST")

    # --- observability
    tracing: TracingMode = Field(default="off", alias="RAG_TRACING")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_format: Literal["text", "json"] = Field(default="text", alias="LOG_FORMAT")

    @field_validator("docs_dir", "index_path", "sql_db_path")
    @classmethod
    def _make_absolute(cls, value: Path) -> Path:
        return _resolve(value)

    @field_validator("anthropic_api_key", "database_url", "api_keys")
    @classmethod
    def _empty_key_is_none(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if self.vector_store == "pgvector" and self.database_url is None:
            raise ValueError("RAG_VECTOR_STORE=pgvector needs RAG_DATABASE_URL")
        return self

    @property
    def llm_enabled(self) -> bool:
        return self.anthropic_api_key is not None

    @property
    def api_key_list(self) -> list[str]:
        if self.api_keys is None:
            return []
        return [key.strip() for key in self.api_keys.get_secret_value().split(",") if key.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
