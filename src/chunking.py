"""Page-aware, chapter-aware, paragraph-first chunking.

Guarantees:
* a chunk never crosses a page boundary (one page == one citation),
* a chunk never crosses a chapter/section boundary,
* every chunk fits the embedding model's token limit (oversized chunks are
  split further; nothing is silently truncated),
* chunk ids are deterministic.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.errors import EmbeddingInputTooLongError
from src.schemas import Chunk, PageText, Segment

def embedding_text(chapter_title: str, section_title: str, text: str) -> str:
    """Text actually embedded: chapter/section title give each chunk its context
    (the stored/cited text is unchanged). Questions are embedded without it."""
    parts = [t for t in (chapter_title, section_title) if t and t != "Chapter unavailable"]
    return (" - ".join(parts) + "\n" + text) if parts else text


SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _splitter(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=min(overlap, max(0, size // 4)),
        separators=SEPARATORS,
        keep_separator="end",
        add_start_index=True,
    )


@dataclass
class ChunkStats:
    oversize_splits: int = 0
    max_tokens_seen: int = 0
    chunks_per_chapter: dict[str, int] = field(default_factory=dict)
    skipped_segments: int = 0


def _hard_split(piece: str, start: int, count_tokens, max_tokens: int) -> list[tuple[str, int]]:
    """Last resort: greedily pack whole words under the token budget (never truncates)."""
    out: list[tuple[str, int]] = []
    cur_s = cur_e = None
    for m in re.finditer(r"\S+", piece):
        if cur_s is not None and count_tokens(piece[cur_s : m.end()]) > max_tokens:
            out.append((piece[cur_s:cur_e], start + cur_s))
            cur_s = None
        if cur_s is None:
            cur_s = m.start()
        cur_e = m.end()
        word = piece[m.start() : m.end()]
        while count_tokens(word) > max_tokens and len(word) > 1:  # one giant "word"
            half = max(1, len(word) // 2)
            out.append((word[:half], start + m.start()))
            word = word[half:]
            cur_s = m.end() - len(word)
    if cur_s is not None:
        out.append((piece[cur_s:cur_e], start + cur_s))
    return [(t, o) for t, o in out if t.strip()]


def _split_to_fit(
    text: str,
    base_offset: int,
    size: int,
    overlap: int,
    count_tokens,
    max_tokens: int,
    stats: ChunkStats,
) -> list[tuple[str, int]]:
    """Split `text`, then recursively re-split any piece over the token limit."""
    out: list[tuple[str, int]] = []
    docs = _splitter(size, overlap).create_documents([text])
    for d in docs:
        piece = d.page_content
        start = base_offset + int(d.metadata["start_index"])
        if not piece.strip():
            continue
        if count_tokens(piece) <= max_tokens:
            out.append((piece, start))
            continue
        stats.oversize_splits += 1
        if size <= 20:
            out.extend(_hard_split(piece, start, count_tokens, max_tokens))
            continue
        out.extend(_split_to_fit(piece, start, size // 2, overlap // 2, count_tokens, max_tokens, stats))
    return out


def _merge_small(pieces, seg, min_chars, max_chars, count_tokens, budget, stats):
    """Merge tiny page-tail fragments into the previous chunk of the same segment.

    The merged text is re-sliced from the segment (exact contiguous source text),
    so offsets/provenance stay exact."""
    if min_chars <= 0 or len(pieces) < 2:
        return pieces
    out = [pieces[0]]
    for text, start in pieces[1:]:
        prev_text, prev_start = out[-1]
        if len(text.strip()) < min_chars:
            lo = prev_start - seg.start
            hi = start - seg.start + len(text)
            merged = seg.text[lo:hi]
            if len(merged) <= max_chars and count_tokens(merged) <= budget:
                out[-1] = (merged, prev_start)
                continue
        out.append((text, start))
    return out


def build_chunks(
    segments: list[Segment],
    *,
    book_id: str,
    source_filename: str,
    chunk_size: int,
    chunk_overlap: int,
    count_tokens,
    max_tokens: int,
    stats: Optional[ChunkStats] = None,
    min_chunk_chars: int = 0,
) -> list[Chunk]:
    stats = stats or ChunkStats()
    chunks: list[Chunk] = []
    per_page_counter: dict[int, int] = {}
    for seg in segments:
        if not seg.indexed:
            stats.skipped_segments += 1
            continue
        prefix_tokens = count_tokens(embedding_text(seg.chapter_title, seg.section_title or "", "")) 
        budget = max(4, max_tokens - prefix_tokens - 2)  # room for the title prefix
        pieces = _split_to_fit(
            seg.text, seg.start, chunk_size, chunk_overlap, count_tokens, budget, stats
        )
        pieces = _merge_small(pieces, seg, min_chunk_chars, int(chunk_size * 1.35), count_tokens, budget, stats)
        for piece, abs_start in pieces:
            text = piece.strip()
            if not any(ch.isalnum() for ch in text):
                continue
            tokens = count_tokens(embedding_text(seg.chapter_title, seg.section_title or "", text))
            if tokens > max_tokens:
                raise EmbeddingInputTooLongError(
                    f"A chunk on PDF page {seg.page_number} has {tokens} tokens (> {max_tokens})."
                )
            n = per_page_counter.get(seg.page_number, 0)
            per_page_counter[seg.page_number] = n + 1
            digest = sha1(text)
            lead = len(piece) - len(piece.lstrip())
            chunks.append(
                Chunk(
                    chunk_id=f"{book_id}-p{seg.page_number:04d}-{n:02d}-{digest[:8]}",
                    chunk_index=len(chunks),
                    book_id=book_id,
                    source_filename=source_filename,
                    text=text,
                    text_hash=digest,
                    pdf_page_index=seg.page_index,
                    pdf_page_number=seg.page_number,
                    printed_page_label=seg.printed_label,
                    chapter_id=seg.chapter_id or "",
                    chapter_title=seg.chapter_title,
                    section_id=seg.section_id or "",
                    section_title=seg.section_title or "",
                    start_char=abs_start + lead,
                    end_char=abs_start + lead + len(text),
                    token_count=tokens,
                )
            )
            stats.max_tokens_seen = max(stats.max_tokens_seen, tokens)
            key = seg.chapter_title
            stats.chunks_per_chapter[key] = stats.chunks_per_chapter.get(key, 0) + 1
    return chunks
