"""Owner-operated ingestion:  python -m src.ingestion

Not reachable from the API or UI. Builds a NEW index, verifies it, then
atomically switches to it. Safe to run repeatedly (idempotent).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import PIPELINE_VERSION, Settings, get_settings
from src.chapter_detection import build_segments, resolve_chapters
from src.chunking import ChunkStats, build_chunks, embedding_text
from src.embeddings import Embedder, get_embedder
from src.errors import BookWiseError, IngestionError
from src.pdf_reader import CONTENT_PAGE_MIN_CHARS, read_pdf
from src.vector_store import ActiveIndex, VectorStore, collection_name, utc_now

log = logging.getLogger("bookwise.ingestion")


def compute_fingerprint(settings: Settings, pdf_sha256: str, mapping_dump: dict, emb_info: dict) -> str:
    """Anything that changes the index content must change this hash."""
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "book_id": settings.book_id,
        "pdf_sha256": pdf_sha256,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "min_chunk_chars": settings.min_chunk_chars,
        "chapter_mapping": mapping_dump,
        "embedding": {
            k: emb_info.get(k)
            for k in ("model", "dimension", "max_seq_length", "sentence_transformers", "preprocess_version")
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def run_ingestion(
    settings: Settings,
    embedder: Optional[Embedder] = None,
    store: Optional[VectorStore] = None,
    *,
    force: bool = False,
    detect_chapters_only: bool = False,
    strict: bool = False,
    prune: bool = True,
) -> dict:
    t0 = time.perf_counter()
    warnings: list[str] = []

    # 1) PDF validation (exists, readable, has extractable text) --------------
    doc = read_pdf(settings.book_path)
    warnings.extend(doc.warnings)

    # 2) Chapter mapping ----------------------------------------------------------
    resolution = resolve_chapters(doc, settings.chapters_path, settings.chapter_outline_depth)
    mapping = resolution.mapping
    warnings.extend(resolution.warnings)
    if strict and not mapping.reviewed:
        raise IngestionError(
            "Strict mode: review data/config/chapters.json and set \"reviewed\": true first."
        )
    segments, seg_warnings = build_segments(doc.pages, mapping)
    warnings.extend(seg_warnings)

    chapter_level = mapping.chapter_level
    chapter_entries = [e for e in mapping.entries if e.level <= chapter_level]
    chapters_report = {
        "source": mapping.source,
        "reviewed": mapping.reviewed,
        "draft_written_to": str(settings.chapters_path) if resolution.wrote_draft else None,
        "entries_total": len(mapping.entries),
        "chapters": len(chapter_entries),
        "sections": len(mapping.entries) - len(chapter_entries),
        "excluded_from_index": [e.title for e in mapping.entries if not e.indexed],
        "needs_owner_review": not mapping.reviewed,
        "entries": [
            {"id": e.id, "title": e.title, "level": e.level, "start_page": e.start_page, "indexed": e.indexed}
            for e in mapping.entries
        ],
    }
    pages_report = {
        "total": doc.page_count,
        "content_pages": sum(1 for p in doc.pages if p.char_count >= CONTENT_PAGE_MIN_CHARS),
        "suspicious_pages": doc.suspicious_pages,
        "printed_page_labels_available": doc.has_reliable_page_labels,
    }

    report: dict = {
        "generated_at": utc_now(),
        "book": {
            "filename": doc.path.name,
            "sha256": doc.sha256,
            "title": settings.book_title_override or doc.title,
            "author": doc.author,
        },
        "pages": pages_report,
        "chapters": chapters_report,
        "boilerplate_removed": doc.removed_boilerplate,
        "warnings": warnings,
    }

    if detect_chapters_only:
        report["status"] = "chapters_drafted_only"
        _write_reports(settings, report)
        return report

    # 3) Embedding model must be available --------------------------------------
    embedder = embedder or get_embedder(settings.embedding_model, settings.embedding_local_only)
    if hasattr(embedder, "check_available"):
        embedder.check_available()
    emb_info = embedder.version_info()

    # 4) Chunking with real token counts ------------------------------------------
    stats = ChunkStats()
    chunks = build_chunks(
        segments,
        book_id=settings.book_id,
        source_filename=doc.path.name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        count_tokens=embedder.count_tokens,
        max_tokens=embedder.max_seq_length,
        stats=stats,
        min_chunk_chars=settings.min_chunk_chars,
    )
    if not chunks:
        raise IngestionError("No chunks were produced; check chapter 'indexed' flags and the PDF text.")
    unavailable = stats.chunks_per_chapter.get("Chapter unavailable", 0)
    if unavailable:
        warnings.append(f"{unavailable} chunk(s) have no chapter ('Chapter unavailable'). Review chapters.json.")

    report["chunks"] = {
        "count": len(chunks),
        "avg_chars": round(statistics.mean(len(c.text) for c in chunks), 1),
        "max_tokens_seen": stats.max_tokens_seen,
        "model_max_tokens": embedder.max_seq_length,
        "oversize_splits": stats.oversize_splits,
        "skipped_excluded_segments": stats.skipped_segments,
        "per_chapter": stats.chunks_per_chapter,
    }
    report["embedding"] = emb_info

    fingerprint = compute_fingerprint(settings, doc.sha256, mapping.model_dump(exclude={"notes"}), emb_info)
    report["fingerprint"] = fingerprint
    store = store or VectorStore(settings.chroma_dir, settings.active_index_path)

    # 5) Idempotency: nothing changed => nothing to do ----------------------------
    try:
        current = store.active() if settings.active_index_path.is_file() else None
    except BookWiseError:
        current = None
    if current and current.fingerprint == fingerprint and not force:
        try:
            healthy = store.count() == current.chunk_count
        except BookWiseError:
            healthy = False
        if healthy:
            report["status"] = "up_to_date"
            report["collection"] = current.collection
            report["timings_s"] = {"total": round(time.perf_counter() - t0, 2)}
            _write_reports(settings, report)
            return report
    if current and current.fingerprint != fingerprint:
        warnings.append(
            "The PDF, chunking settings, chapter mapping or embedding configuration changed; "
            "a replacement index is being built before the old one is retired."
        )

    # 6) Embed + build new collection + verify + activate ----------------------
    t_embed = time.perf_counter()
    embeddings = embedder.embed_documents(
        [embedding_text(c.chapter_title, c.section_title, c.text) for c in chunks]
    )
    if len(embeddings) != len(chunks) or any(len(v) != emb_info["dimension"] for v in embeddings):
        raise IngestionError("Embedding output shape mismatch.")
    t_embed = time.perf_counter() - t_embed

    name = store.build_collection(fingerprint, chunks, embeddings, emb_info["model"], emb_info["dimension"])
    if not store.verify_self_retrieval(name, chunks[0], embeddings[0]):
        raise IngestionError("Index self-check failed (first chunk was not its own nearest neighbour). Not activated.")

    store.activate(
        ActiveIndex(
            fingerprint=fingerprint,
            collection=name,
            book_title=settings.book_title_override or doc.title or "Selected book",
            author=doc.author,
            page_count=doc.page_count,
            chunk_count=len(chunks),
            embedding_model=emb_info["model"],
            embedding_dimension=emb_info["dimension"],
            indexed_at=utc_now(),
            source_sha256=doc.sha256,
        )
    )
    removed = store.prune_other_collections(name) if prune else []
    report["status"] = "indexed"
    report["collection"] = name
    report["retired_collections"] = removed
    report["timings_s"] = {"embedding": round(t_embed, 2), "total": round(time.perf_counter() - t0, 2)}
    _write_reports(settings, report)
    return report


# ------------------------------------------------------------------ reporting
def _write_reports(settings: Settings, report: dict) -> None:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "ingestion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (settings.reports_dir / "ingestion_report.md").write_text(render_markdown(report), encoding="utf-8")


def render_markdown(r: dict) -> str:
    lines = [f"# Ingestion report ({r['generated_at']})", "", f"**Status:** {r.get('status')}", ""]
    b = r["book"]
    lines += [f"- Book: {b.get('title')} ({b['filename']}), author: {b.get('author') or 'n/a'}", f"- SHA-256: `{b['sha256']}`"]
    p = r["pages"]
    lines += [
        f"- Pages: {p['total']} (content pages: {p['content_pages']})",
        f"- Empty/suspicious pages (PDF numbers): {p['suspicious_pages'] or 'none'}",
        f"- Reliable printed page labels in PDF: {p['printed_page_labels_available']}",
        "",
        "## Chapters",
    ]
    c = r["chapters"]
    lines += [
        f"- Source: {c['source']}; owner-reviewed: {c['reviewed']}",
        f"- Entries: {c['entries_total']} ({c['chapters']} chapters, {c['sections']} sections)",
        f"- Excluded from index: {c['excluded_from_index'] or 'none'}",
        "",
    ]
    for e in c["entries"]:
        flag = "" if e["indexed"] else "  _(excluded)_"
        lines.append(f"{'  ' * e['level']}- p.{e['start_page']}: {e['title']}{flag}")
    if "chunks" in r:
        k = r["chunks"]
        lines += [
            "",
            "## Chunks",
            f"- Count: {k['count']}, average chars: {k['avg_chars']}",
            f"- Max tokens seen: {k['max_tokens_seen']} (model limit {k['model_max_tokens']}), oversize splits: {k['oversize_splits']}",
        ]
        e = r["embedding"]
        lines += [
            "",
            "## Embedding",
            f"- Model: {e['model']}, dimension: {e['dimension']}, max_seq_length: {e['max_seq_length']}",
            f"- sentence-transformers: {e['sentence_transformers']}, normalised: {e['normalized']}",
        ]
    if r.get("boilerplate_removed"):
        lines += ["", "## Removed repeated headers/footers", json.dumps(r["boilerplate_removed"], ensure_ascii=False)]
    lines += ["", "## Warnings"] + ([f"- {w}" for w in r["warnings"]] or ["- none"])
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Owner-only: index the fixed book into ChromaDB.")
    ap.add_argument("--force", action="store_true", help="Rebuild even if nothing changed.")
    ap.add_argument("--detect-chapters-only", action="store_true", help="Write a draft chapters.json and stop.")
    ap.add_argument("--strict", action="store_true", help="Refuse to index unless chapters.json has reviewed=true.")
    ap.add_argument("--keep-old", action="store_true", help="Do not delete the previous index collection.")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    try:
        report = run_ingestion(
            settings,
            force=args.force,
            detect_chapters_only=args.detect_chapters_only,
            strict=args.strict,
            prune=not args.keep_old,
        )
    except BookWiseError as exc:
        print(f"\nERROR [{exc.code}]: {exc.detail}", file=sys.stderr)
        return 1
    print(f"\nStatus: {report['status']}")
    print(f"Report: {settings.reports_dir / 'ingestion_report.md'}")
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    if report["status"] == "chapters_drafted_only":
        print(f"Review {settings.chapters_path}, set \"reviewed\": true, then run ingestion again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
