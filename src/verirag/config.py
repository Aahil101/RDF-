"""Central configuration for VeriRAG.

Settings are resolved with the precedence:

    explicit constructor argument  >  environment variable  >  built-in default

A ``.env`` file in the project root is loaded automatically when
``python-dotenv`` is installed, so local development needs no shell exports.
Every setting has a working default: VeriRAG is runnable with zero
configuration and zero API keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

try:  # optional dependency, gracefully skipped
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only without python-dotenv

    def load_dotenv(*_args, **_kwargs) -> bool:  # type: ignore[misc]
        return False


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env_str(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env_str(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    return _env_str(key, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """Runtime settings for the whole pipeline."""

    # ---------------------------------------------------------------- paths
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _env_str("VERIRAG_DATA_DIR", "data"))

    # ------------------------------------------------------------ ingestion
    chunk_target_words: int = field(default_factory=lambda: _env_int("VERIRAG_CHUNK_TARGET_WORDS", 140))
    chunk_overlap_lines: int = field(default_factory=lambda: _env_int("VERIRAG_CHUNK_OVERLAP_LINES", 2))
    min_chunk_words: int = field(default_factory=lambda: _env_int("VERIRAG_MIN_CHUNK_WORDS", 12))

    # ------------------------------------------------------------- indexing
    embedder: str = field(default_factory=lambda: _env_str("VERIRAG_EMBEDDER", "hashing"))
    embed_model: str = field(
        default_factory=lambda: _env_str("VERIRAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embed_dim: int = field(default_factory=lambda: _env_int("VERIRAG_EMBED_DIM", 384))
    vector_backend: str = field(default_factory=lambda: _env_str("VERIRAG_VECTOR_BACKEND", "numpy"))

    # ------------------------------------------------------------ retrieval
    top_k_dense: int = field(default_factory=lambda: _env_int("VERIRAG_TOP_K_DENSE", 20))
    top_k_lexical: int = field(default_factory=lambda: _env_int("VERIRAG_TOP_K_LEXICAL", 20))
    top_k_final: int = field(default_factory=lambda: _env_int("VERIRAG_TOP_K_FINAL", 5))
    rrf_k: int = field(default_factory=lambda: _env_int("VERIRAG_RRF_K", 60))
    reranker: str = field(default_factory=lambda: _env_str("VERIRAG_RERANKER", "lexical"))
    rerank_model: str = field(
        default_factory=lambda: _env_str("VERIRAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    multi_query: bool = field(default_factory=lambda: _env_bool("VERIRAG_MULTI_QUERY", True))
    # Hard refusal gate on the reranker's top score. Deliberately low: measured
    # on the sample corpus (`verirag eval`), a keyword embedder cannot separate
    # out-of-domain questions from weakly-phrased real ones by score alone, and
    # wrongly refusing a real question is the worse failure. Semantic refusal is
    # delegated to the LLM's INSUFFICIENT_CONTEXT contract; scores below
    # `low_confidence_score` are answered but flagged as weak evidence.
    min_retrieval_score: float = field(default_factory=lambda: _env_float("VERIRAG_MIN_RETRIEVAL_SCORE", 0.10))
    low_confidence_score: float = field(default_factory=lambda: _env_float("VERIRAG_LOW_CONFIDENCE_SCORE", 0.35))

    # ----------------------------------------------------------- generation
    llm_provider: str = field(default_factory=lambda: _env_str("VERIRAG_LLM_PROVIDER", "auto"))
    groq_api_key: str = field(default_factory=lambda: _env_str("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: _env_str("GROQ_MODEL", "openai/gpt-oss-120b"))
    gemini_api_key: str = field(default_factory=lambda: _env_str("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: _env_str("GEMINI_MODEL", "gemini-2.0-flash"))
    ollama_base_url: str = field(default_factory=lambda: _env_str("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env_str("OLLAMA_MODEL", "llama3.2"))
    temperature: float = field(default_factory=lambda: _env_float("VERIRAG_TEMPERATURE", 0.1))
    max_tokens: int = field(default_factory=lambda: _env_int("VERIRAG_MAX_TOKENS", 900))
    request_timeout: int = field(default_factory=lambda: _env_int("VERIRAG_REQUEST_TIMEOUT", 60))

    # ------------------------------------------------------------ grounding
    grounding_threshold: float = field(default_factory=lambda: _env_float("VERIRAG_GROUNDING_THRESHOLD", 0.42))

    # ----------------------------------------------------------------- proof
    highlight_dpi: int = field(default_factory=lambda: _env_int("VERIRAG_HIGHLIGHT_DPI", 140))
    highlight_rgb: tuple[float, float, float] = (1.0, 0.85, 0.15)

    # ------------------------------------------------------------ derived paths
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def proof_dir(self) -> Path:
        return self.data_dir / "proof_cache"

    @property
    def chat_db_path(self) -> Path:
        return self.index_dir / "verirag.sqlite3"

    def ensure_dirs(self) -> "Settings":
        """Create every directory the pipeline writes to."""
        for path in (self.raw_dir, self.upload_dir, self.index_dir, self.proof_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    @staticmethod
    def is_explicitly_set(*env_keys: str) -> bool:
        """True if the user set any of *env_keys* themselves.

        Used so auto-calibration never silently overrides a deliberate choice.
        """
        return any((os.getenv(key) or "").strip() != "" for key in env_keys)


_SETTINGS: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    global _SETTINGS
    if _SETTINGS is None or refresh:
        _SETTINGS = Settings().ensure_dirs()
    return _SETTINGS
