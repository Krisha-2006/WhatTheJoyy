"""Local SentenceTransformers embeddings.

The SAME `preprocess()` and model are used for book chunks and user questions.
"""
from __future__ import annotations

import threading
from typing import Protocol

from src.errors import EmbeddingUnavailableError

PREPROCESS_VERSION = "1"


def preprocess(text: str) -> str:
    """Whitespace-normalise only. Case and punctuation are left to the model."""
    return " ".join(text.split())


class Embedder(Protocol):
    model_name: str

    @property
    def dimension(self) -> int: ...
    @property
    def max_seq_length(self) -> int: ...
    def count_tokens(self, text: str) -> int: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def version_info(self) -> dict: ...


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers. The model is loaded lazily and once."""

    def __init__(self, model_name: str, local_only: bool = False):
        self.model_name = model_name
        self.local_only = local_only
        self._model = None
        self._lock = threading.Lock()

    # -- loading ------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, local_files_only=self.local_only)
            except Exception as exc:  # noqa: BLE001 - network/disk/import problems
                raise EmbeddingUnavailableError(
                    f"Could not load embedding model {self.model_name!r}: {exc}. "
                    "Run once with internet access to download it, or check EMBEDDING_MODEL / "
                    "EMBEDDING_LOCAL_ONLY."
                ) from exc
        return self._model

    def check_available(self) -> None:
        self._load()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- properties -----------------------------------------------------------
    @property
    def dimension(self) -> int:
        m = self._load()
        getter = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        return int(getter())

    @property
    def max_seq_length(self) -> int:
        return int(self._load().max_seq_length)

    def count_tokens(self, text: str) -> int:
        tok = self._load().tokenizer
        return len(tok(preprocess(text), add_special_tokens=True, truncation=False)["input_ids"])

    # -- inference ------------------------------------------------------------
    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        try:
            vecs = model.encode(
                [preprocess(t) for t in texts],
                batch_size=32,
                normalize_embeddings=True,  # unit vectors -> cosine == dot product
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingUnavailableError(f"Embedding inference failed: {exc}") from exc
        return [v.tolist() for v in vecs]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def version_info(self) -> dict:
        try:
            import sentence_transformers

            st_version = sentence_transformers.__version__
        except Exception:  # noqa: BLE001
            st_version = "unknown"
        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": self.max_seq_length,
            "sentence_transformers": st_version,
            "preprocess_version": PREPROCESS_VERSION,
            "normalized": True,
        }


_cached: dict[tuple[str, bool], SentenceTransformerEmbedder] = {}


def get_embedder(model_name: str, local_only: bool = False) -> SentenceTransformerEmbedder:
    key = (model_name, local_only)
    if key not in _cached:
        _cached[key] = SentenceTransformerEmbedder(model_name, local_only)
    return _cached[key]
