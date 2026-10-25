"""Persistent ChromaDB wrapper with safe, versioned index activation.

Similarity metric
-----------------
Collections are created with cosine space. Chroma then reports a cosine
DISTANCE d = 1 - cos(a, b), in [0, 2]. We convert once, here:

    similarity = 1 - distance          (1.0 identical, 0 unrelated)

Everything downstream (thresholds, logs, docs) talks about SIMILARITY only.

Safe re-indexing
----------------
Each index lives in its own collection `bookwise_<fingerprint16>`. Ingestion
builds the new collection completely, verifies it, and only then atomically
rewrites runtime/active_index.json. The previous working collection is deleted
only after that switch.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.errors import IndexNotReadyError, VectorStoreError
from src.schemas import Chunk, RetrievedChunk

COLLECTION_PREFIX = "bookwise_"


@dataclass
class ActiveIndex:
    fingerprint: str
    collection: str
    book_title: str
    author: Optional[str]
    page_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimension: int
    indexed_at: str
    source_sha256: str


def collection_name(fingerprint: str) -> str:
    return f"{COLLECTION_PREFIX}{fingerprint[:16]}"


def read_active_index(path: Path) -> Optional[ActiveIndex]:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return ActiveIndex(**json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        raise IndexNotReadyError(f"active_index.json is unreadable: {exc}") from exc


def write_active_index(path: Path, active: ActiveIndex) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(active.__dict__, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic on POSIX and Windows


def _to_meta(c: Chunk, fingerprint: str, model: str, dim: int) -> dict:
    # Chroma metadata values must be str/int/float/bool (no None).
    return {
        "book_id": c.book_id,
        "source_filename": c.source_filename,
        "chunk_id": c.chunk_id,
        "chunk_index": c.chunk_index,
        "text_hash": c.text_hash,
        "pdf_page_index": c.pdf_page_index,
        "pdf_page_number": c.pdf_page_number,
        "printed_page_label": c.printed_page_label or "",
        "chapter_id": c.chapter_id,
        "chapter_title": c.chapter_title,
        "section_id": c.section_id,
        "section_title": c.section_title,
        "start_char": c.start_char,
        "end_char": c.end_char,
        "token_count": c.token_count,
        "embedding_model": model,
        "embedding_dimension": dim,
        "index_fingerprint": fingerprint,
    }


def _to_retrieved(doc: str, meta: dict, similarity: Optional[float], neighbor: bool = False) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=meta["chunk_id"],
        chunk_index=int(meta["chunk_index"]),
        text=doc,
        chapter_id=meta.get("chapter_id", ""),
        chapter_title=meta.get("chapter_title", ""),
        section_title=meta.get("section_title", ""),
        pdf_page_index=int(meta["pdf_page_index"]),
        pdf_page_number=int(meta["pdf_page_number"]),
        printed_page_label=(meta.get("printed_page_label") or None),
        start_char=int(meta.get("start_char", -1)),
        end_char=int(meta.get("end_char", -1)),
        similarity=similarity,
        is_neighbor=neighbor,
    )


class VectorStore:
    def __init__(self, chroma_dir: Path, active_index_path: Path):
        self.chroma_dir = Path(chroma_dir)
        self.active_index_path = Path(active_index_path)
        self._client = None
        self._lock = threading.Lock()
        self._cached_fingerprint: Optional[str] = None
        self._all_cache: tuple[Optional[str], list] = (None, [])

    # ------------------------------------------------------------ client
    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                self.chroma_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self.chroma_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Could not open ChromaDB at {self.chroma_dir}: {exc}") from exc
        return self._client

    def reset_client(self) -> None:
        self._client = None

    # ------------------------------------------------------------ building
    def build_collection(
        self,
        fingerprint: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        model: str,
        dim: int,
    ) -> str:
        name = collection_name(fingerprint)
        client = self._get_client()
        try:
            try:  # a half-built leftover from a crashed run is safe to discard
                client.delete_collection(name)
            except Exception:  # noqa: BLE001
                pass
            col = client.create_collection(
                name=name,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"description": "BookWise chunks", "similarity": "cosine"},
                embedding_function=None,  # we always supply our own vectors
            )
            batch = max(1, min(500, client.get_max_batch_size()))
            for i in range(0, len(chunks), batch):
                part = chunks[i : i + batch]
                col.upsert(
                    ids=[c.chunk_id for c in part],
                    documents=[c.text for c in part],
                    embeddings=embeddings[i : i + batch],
                    metadatas=[_to_meta(c, fingerprint, model, dim) for c in part],
                )
            if col.count() != len(chunks):
                raise VectorStoreError(
                    f"Index verification failed: expected {len(chunks)} chunks, found {col.count()}."
                )
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to build the ChromaDB collection: {exc}") from exc
        return name

    def verify_self_retrieval(self, name: str, probe_chunk: Chunk, probe_embedding: list[float]) -> bool:
        col = self._get_client().get_collection(name, embedding_function=None)
        res = col.query(query_embeddings=[probe_embedding], n_results=1, include=["metadatas"])
        return bool(res["ids"] and res["ids"][0] and res["ids"][0][0] == probe_chunk.chunk_id)

    def activate(self, active: ActiveIndex) -> None:
        write_active_index(self.active_index_path, active)
        self._cached_fingerprint = None

    def prune_other_collections(self, keep_name: str) -> list[str]:
        client = self._get_client()
        removed = []
        for col in client.list_collections():
            name = col.name if hasattr(col, "name") else str(col)
            if name.startswith(COLLECTION_PREFIX) and name != keep_name:
                client.delete_collection(name)
                removed.append(name)
        return removed

    # ------------------------------------------------------------ reading
    def active(self) -> ActiveIndex:
        active = read_active_index(self.active_index_path)
        if active is None:
            raise IndexNotReadyError("No active index. Run: python -m src.ingestion")
        return active

    def _collection(self):
        active = self.active()
        with self._lock:
            # If ingestion activated a new index, reopen the client so it sees it.
            if self._cached_fingerprint not in (None, active.fingerprint):
                self.reset_client()
            self._cached_fingerprint = active.fingerprint
        try:
            return self._get_client().get_collection(active.collection, embedding_function=None)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Active collection {active.collection!r} cannot be opened: {exc}") from exc

    def count(self) -> int:
        try:
            return self._collection().count()
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(str(exc)) from exc

    def query(self, embedding: list[float], n: int) -> list[RetrievedChunk]:
        col = self._collection()
        try:
            res = col.query(
                query_embeddings=[embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Vector search failed: {exc}") from exc
        out: list[RetrievedChunk] = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append(_to_retrieved(doc, meta, similarity=1.0 - float(dist)))
        return out

    def all_chunks(self) -> list[RetrievedChunk]:
        """Every chunk of the active index in reading order (cached per index).
        Used to build the in-memory keyword (BM25) index."""
        active = self.active()
        if self._all_cache[0] == active.fingerprint:
            return self._all_cache[1]
        col = self._collection()
        try:
            res = col.get(include=["documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not read chunks: {exc}") from exc
        chunks = [_to_retrieved(d, m, None) for d, m in zip(res["documents"], res["metadatas"])]
        chunks.sort(key=lambda c: c.chunk_index)
        self._all_cache = (active.fingerprint, chunks)
        return chunks

    def get_by_chunk_index(self, indexes: list[int]) -> list[RetrievedChunk]:
        if not indexes:
            return []
        col = self._collection()
        try:
            res = col.get(where={"chunk_index": {"$in": indexes}}, include=["documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Neighbour lookup failed: {exc}") from exc
        return [_to_retrieved(d, m, None, neighbor=True) for d, m in zip(res["documents"], res["metadatas"])]

    def ping(self) -> bool:
        try:
            self._collection()
            return True
        except Exception:  # noqa: BLE001
            return False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
