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
    top_k: int = Field(default=4, ge=1, le=20, alias="RAG_TOP_K")
    min_score: float = Field(default=0.05, ge=0.0, le=1.0, alias="RAG_MIN_SCORE")

    # --- generation
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-opus-5", alias="ANTHROPIC_MODEL")
    judge_model: str = Field(default="claude-sonnet-5", alias="RAG_JUDGE_MODEL")
    max_output_tokens: int = Field(default=2048, gt=0, alias="RAG_MAX_OUTPUT_TOKENS")
    llm_timeout_seconds: float = Field(default=60.0, gt=0, alias="RAG_LLM_TIMEOUT_SECONDS")

    # --- observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_format: Literal["text", "json"] = Field(default="text", alias="LOG_FORMAT")

    @field_validator("docs_dir", "index_path")
    @classmethod
    def _make_absolute(cls, value: Path) -> Path:
        return _resolve(value)

    @field_validator("anthropic_api_key")
    @classmethod
    def _empty_key_is_none(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        return self

    @property
    def llm_enabled(self) -> bool:
        return self.anthropic_api_key is not None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
