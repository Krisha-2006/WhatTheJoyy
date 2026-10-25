"""Central configuration for BookWise.

Every tunable value lives here and can be overridden with an environment
variable (or a `.env` file). See `.env.example` for the full list.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Bump when chunking/cleaning/preprocessing logic changes in a way that should
# force a re-index even if the PDF and settings are unchanged.
PIPELINE_VERSION = "2"

FALLBACK_TEXT = "The selected book does not provide enough information to answer this question."


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    p = Path(value) if value else default
    return p if p.is_absolute() else (ROOT / p)


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Fixed book -------------------------------------------------------
    book_id: str = "selected_book"
    book_path: Path = ROOT / "data" / "books" / "selected_book.pdf"
    chapters_path: Path = ROOT / "data" / "config" / "chapters.json"
    book_title_override: str = ""
    # Outline depth that counts as a "chapter" (0 = top-level bookmarks).
    # Deeper bookmarks are reported as "sections" inside the chapter.
    chapter_outline_depth: int = 0

    # --- Storage ----------------------------------------------------------
    chroma_dir: Path = ROOT / "chroma_db"
    runtime_dir: Path = ROOT / "runtime"

    # --- Embeddings -------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # True = never touch the network (model must already be cached).
    embedding_local_only: bool = False

    # --- Chunking (starting values, not universal best settings) ---------
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # Page-tail fragments shorter than this are merged into the previous chunk.
    min_chunk_chars: int = 300

    # --- Retrieval policy (calibrate on YOUR book: python -m src.diagnose / calibrate)
    candidate_k: int = 20
    final_k: int = 4
    # Dense gates use cosine SIMILARITY (= 1 - cosine distance). Starting values only.
    min_similarity: float = 0.20
    min_top_similarity: float = 0.28
    # Hybrid retrieval: BM25-style keyword search fused with dense search (RRF).
    hybrid_enabled: bool = True
    # IDF-weighted share of the question's terms found in a chunk (0..1).
    lexical_min_coverage: float = 0.50
    lexical_top_coverage: float = 0.60
    neighbor_expansion: int = 0  # 0 = off; 1 = add the chunk before/after
    max_context_chars: int = 4000

    # --- LLM (Ollama) -----------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_num_ctx: int = 4096
    ollama_timeout_seconds: float = 120.0
    ollama_keep_alive: str = "30m"
    ollama_num_predict: int = 400
    # off = deterministic checks only; single = ONE extra LLM review call (default);
    # per_claim = one call per claim + one judge call (slowest, strictest)
    llm_verifier_mode: str = "single"
    grounding_min_lexical_overlap: float = 0.4
    warmup_on_start: bool = True

    # --- Auth / API -------------------------------------------------------
    secret_key: str = ""
    session_ttl_hours: int = 12
    allow_signup: bool = True
    invite_code: str = ""
    min_password_length: int = 10
    max_question_chars: int = 500
    rate_limit_per_minute: int = 6
    login_attempts_per_minute: int = 8

    # --- UI ---------------------------------------------------------------
    api_base_url: str = "http://localhost:8000"

    @property
    def auth_db_path(self) -> Path:
        return self.runtime_dir / "auth.sqlite3"

    @property
    def active_index_path(self) -> Path:
        return self.runtime_dir / "active_index.json"

    @property
    def reports_dir(self) -> Path:
        return self.runtime_dir / "reports"

    def validate_for_api(self) -> None:
        from src.errors import ConfigError

        if len(self.secret_key) < 32 or self.secret_key.lower().startswith("change"):
            raise ConfigError(
                "SECRET_KEY is missing, too short (<32 chars) or still the placeholder. "
                'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )



def _verifier_mode() -> str:
    mode = _str("LLM_VERIFIER_MODE", "").lower()
    if not mode:  # legacy switch from v1
        mode = "off" if not _bool("LLM_VERIFIER_ENABLED", True) else "single"
    if mode not in {"off", "single", "per_claim"}:
        raise ValueError("LLM_VERIFIER_MODE must be off, single or per_claim")
    return mode


def load_settings() -> Settings:
    return Settings(
        book_id=_str("BOOK_ID", "selected_book"),
        book_path=_path("BOOK_PDF_PATH", ROOT / "data" / "books" / "selected_book.pdf"),
        chapters_path=_path("CHAPTERS_PATH", ROOT / "data" / "config" / "chapters.json"),
        book_title_override=_str("BOOK_TITLE", ""),
        chapter_outline_depth=_int("CHAPTER_OUTLINE_DEPTH", 0),
        chroma_dir=_path("CHROMA_DIR", ROOT / "chroma_db"),
        runtime_dir=_path("RUNTIME_DIR", ROOT / "runtime"),
        embedding_model=_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_local_only=_bool("EMBEDDING_LOCAL_ONLY", False),
        chunk_size=_int("CHUNK_SIZE", 1000),
        chunk_overlap=_int("CHUNK_OVERLAP", 150),
        min_chunk_chars=_int("MIN_CHUNK_CHARS", 300),
        candidate_k=_int("CANDIDATE_K", 20),
        final_k=_int("FINAL_K", 4),
        min_similarity=_float("MIN_SIMILARITY", 0.20),
        min_top_similarity=_float("MIN_TOP_SIMILARITY", 0.28),
        hybrid_enabled=_bool("HYBRID_ENABLED", True),
        lexical_min_coverage=_float("LEXICAL_MIN_COVERAGE", 0.50),
        lexical_top_coverage=_float("LEXICAL_TOP_COVERAGE", 0.60),
        neighbor_expansion=_int("NEIGHBOR_EXPANSION", 0),
        max_context_chars=_int("MAX_CONTEXT_CHARS", 4000),
        ollama_base_url=_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_str("OLLAMA_MODEL", "llama3.2:3b"),
        ollama_num_ctx=_int("OLLAMA_NUM_CTX", 4096),
        ollama_timeout_seconds=_float("OLLAMA_TIMEOUT_SECONDS", 120.0),
        ollama_keep_alive=_str("OLLAMA_KEEP_ALIVE", "30m"),
        ollama_num_predict=_int("OLLAMA_NUM_PREDICT", 400),
        llm_verifier_mode=_verifier_mode(),
        grounding_min_lexical_overlap=_float("GROUNDING_MIN_LEXICAL_OVERLAP", 0.4),
        warmup_on_start=_bool("WARMUP_ON_START", True),
        secret_key=_str("SECRET_KEY", ""),
        session_ttl_hours=_int("SESSION_TTL_HOURS", 12),
        allow_signup=_bool("ALLOW_SIGNUP", True),
        invite_code=_str("INVITE_CODE", ""),
        min_password_length=_int("MIN_PASSWORD_LENGTH", 10),
        max_question_chars=_int("MAX_QUESTION_CHARS", 500),
        rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 6),
        login_attempts_per_minute=_int("LOGIN_ATTEMPTS_PER_MINUTE", 8),
        api_base_url=_str("API_BASE_URL", "http://localhost:8000"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
