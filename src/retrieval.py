"""Hybrid retrieval: dense (embeddings) + keyword (BM25) -> fusion -> relevance
policy -> dedupe -> bounded context.

Relevance policy (all values configurable, see .env.example)
-----------------------------------------------------------
* The question is rejected unless EITHER the best dense similarity >=
  MIN_TOP_SIMILARITY OR the best keyword coverage >= LEXICAL_TOP_COVERAGE.
* A chunk is kept if its dense similarity >= MIN_SIMILARITY OR its keyword
  coverage >= LEXICAL_MIN_COVERAGE.
* Kept chunks are ordered by fused rank (Reciprocal Rank Fusion).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Optional

from config import Settings
from src.embeddings import Embedder
from src.lexical import BM25Index
from src.chunking import embedding_text
from src.schemas import Evidence, RetrievalResult, RetrievedChunk
from src.vector_store import VectorStore

RRF_K = 60


@dataclass(frozen=True)
class RetrievalPolicy:
    candidate_k: int = 20
    final_k: int = 4
    min_similarity: float = 0.20
    min_top_similarity: float = 0.28
    hybrid_enabled: bool = True
    lexical_min_coverage: float = 0.50
    lexical_top_coverage: float = 0.60
    neighbor_expansion: int = 0
    max_context_chars: int = 4000

    @classmethod
    def from_settings(cls, s: Settings) -> "RetrievalPolicy":
        return cls(
            candidate_k=s.candidate_k,
            final_k=s.final_k,
            min_similarity=s.min_similarity,
            min_top_similarity=s.min_top_similarity,
            hybrid_enabled=s.hybrid_enabled,
            lexical_min_coverage=s.lexical_min_coverage,
            lexical_top_coverage=s.lexical_top_coverage,
            neighbor_expansion=s.neighbor_expansion,
            max_context_chars=s.max_context_chars,
        )


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _overlaps(a: RetrievedChunk, b: RetrievedChunk) -> bool:
    """True if two chunks of the same page share more than half of the shorter one."""
    if a.pdf_page_number != b.pdf_page_number or a.start_char < 0 or b.start_char < 0:
        return False
    inter = min(a.end_char, b.end_char) - max(a.start_char, b.start_char)
    shorter = min(a.end_char - a.start_char, b.end_char - b.start_char)
    return shorter > 0 and inter / shorter > 0.5


def fuse(dense: list[RetrievedChunk], lexical: list[tuple[RetrievedChunk, float]]) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion. `lexical` items are (chunk, coverage)."""
    by_id: dict[str, RetrievedChunk] = {}
    for rank, c in enumerate(dense, start=1):
        by_id[c.chunk_id] = replace(c, score=1.0 / (RRF_K + rank))
    for rank, (c, cov) in enumerate(lexical, start=1):
        add = 1.0 / (RRF_K + rank)
        if c.chunk_id in by_id:
            cur = by_id[c.chunk_id]
            by_id[c.chunk_id] = replace(cur, score=cur.score + add, lex_coverage=cov)
        else:
            by_id[c.chunk_id] = replace(c, score=add, lex_coverage=cov, similarity=None)
    return sorted(by_id.values(), key=lambda c: c.score, reverse=True)


def filter_candidates(candidates: list[RetrievedChunk], policy: RetrievalPolicy) -> tuple[list[RetrievedChunk], Optional[str]]:
    """Apply the relevance policy to fused candidates. Returns (kept, rejection_reason)."""
    if not candidates:
        return [], "no_candidates"
    top_dense = max((c.similarity or 0.0 for c in candidates), default=0.0)
    top_lex = max((c.lex_coverage or 0.0 for c in candidates), default=0.0)
    if top_dense < policy.min_top_similarity and top_lex < policy.lexical_top_coverage:
        return [], (
            f"weak_evidence(dense={top_dense:.3f}<{policy.min_top_similarity}, "
            f"keywords={top_lex:.2f}<{policy.lexical_top_coverage})"
        )
    kept = [
        c
        for c in candidates
        if (c.similarity or 0.0) >= policy.min_similarity
        or (c.lex_coverage or 0.0) >= policy.lexical_min_coverage
    ]
    if not kept:
        return [], "no_chunk_above_floor"
    return kept, None


def dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    seen_text: set[str] = set()
    for c in chunks:  # input is ordered best-first, so the better duplicate wins
        key = _norm(c.text)
        if key in seen_text or any(_overlaps(c, o) for o in out):
            continue
        seen_text.add(key)
        out.append(c)
    return out


def assemble_context(selected: list[RetrievedChunk], neighbors: list[RetrievedChunk], max_chars: int) -> list[RetrievedChunk]:
    """Add whole chunks (never truncated) best-first until the budget is hit."""
    chosen: list[RetrievedChunk] = []
    used = 0
    for c in selected + neighbors:
        if chosen and used + len(c.text) > max_chars:
            continue
        chosen.append(c)
        used += len(c.text)
    chosen.sort(key=lambda c: c.chunk_index)  # reading order
    return chosen


class Retriever:
    def __init__(self, policy: RetrievalPolicy, embedder: Embedder, store: VectorStore):
        self.policy = policy
        self.embedder = embedder
        self.store = store
        self._lex: Optional[BM25Index] = None
        self._lex_chunks: list[RetrievedChunk] = []
        self._lex_fp: Optional[str] = None
        self._lock = threading.Lock()
        self.last_timings: dict[str, float] = {}

    def _lexical_index(self) -> tuple[BM25Index, list[RetrievedChunk]]:
        fp = self.store.active().fingerprint
        with self._lock:
            if self._lex is None or self._lex_fp != fp:
                chunks = self.store.all_chunks()
                docs = [embedding_text(c.chapter_title, c.section_title, c.text) for c in chunks]
                self._lex, self._lex_chunks, self._lex_fp = BM25Index(docs), chunks, fp
        return self._lex, self._lex_chunks

    def warmup(self) -> None:
        self.embedder.embed_query("warm up")
        if self.policy.hybrid_enabled:
            self._lexical_index()

    def candidates(self, question: str) -> list[RetrievedChunk]:
        """Fused dense+keyword candidates, best first (no policy applied)."""
        p = self.policy
        t = time.perf_counter()
        qvec = self.embedder.embed_query(question)
        t_embed = time.perf_counter() - t
        t = time.perf_counter()
        dense = self.store.query(qvec, max(p.candidate_k, 30))
        lexical: list[tuple[RetrievedChunk, float]] = []
        if p.hybrid_enabled:
            index, chunks = self._lexical_index()
            lexical = [(chunks[h.index], h.coverage) for h in index.search(question, p.candidate_k)]
        self.last_timings = {"embed_s": round(t_embed, 3), "search_s": round(time.perf_counter() - t, 3)}
        return fuse(dense[: max(p.candidate_k, 30)], lexical)

    def retrieve(self, question: str) -> RetrievalResult:
        p = self.policy
        fused = self.candidates(question)[: max(p.candidate_k, 30)]
        top = max((c.similarity or 0.0 for c in fused), default=None)

        kept, reason = filter_candidates(fused, p)
        if reason:
            return RetrievalResult([], top, len(fused), rejected_reason=reason)

        selected = dedupe(kept)[: p.final_k]

        neighbors: list[RetrievedChunk] = []
        if p.neighbor_expansion > 0:
            have = {c.chunk_index for c in selected}
            wanted: list[int] = []
            for c in selected:
                for d in range(1, p.neighbor_expansion + 1):
                    for idx in (c.chunk_index - d, c.chunk_index + d):
                        if idx >= 0 and idx not in have and idx not in wanted:
                            wanted.append(idx)
            same_chapter = {c.chapter_id for c in selected}
            neighbors = [n for n in self.store.get_by_chunk_index(wanted) if n.chapter_id in same_chapter]
            neighbors.sort(key=lambda n: n.chunk_index)

        context = assemble_context(selected, neighbors, p.max_context_chars)
        evidence = [Evidence(label=f"E{i}", chunk=c) for i, c in enumerate(context, start=1)]
        return RetrievalResult(evidence, top, len(fused))
