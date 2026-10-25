import json
from dataclasses import replace

import pytest
from reportlab.pdfgen import canvas

from src.errors import (
    EmbeddingUnavailableError,
    IngestionError,
    NoExtractableTextError,
    PdfMissingError,
    PdfUnreadableError,
)
from src.embeddings import SentenceTransformerEmbedder
from src.ingestion import run_ingestion
from src.vector_store import VectorStore, read_active_index
from tests.conftest import FakeEmbedder


def test_missing_pdf(make_settings, embedder, tmp_path):
    s = make_settings(book_path=tmp_path / "nope.pdf")
    with pytest.raises(PdfMissingError, match="selected_book.pdf"):
        run_ingestion(s, embedder)


def test_corrupt_pdf(make_settings, embedder, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\nthis is definitely not a real pdf \x00\x01\x02")
    with pytest.raises(PdfUnreadableError):
        run_ingestion(make_settings(book_path=bad), embedder)


def test_pdf_without_extractable_text_is_rejected(make_settings, embedder, tmp_path):
    p = tmp_path / "scanned.pdf"
    c = canvas.Canvas(str(p))
    for _ in range(10):  # blank pages = what an image-only scan looks like to pypdf
        c.showPage()
    c.save()
    with pytest.raises(NoExtractableTextError, match="scanned"):
        run_ingestion(make_settings(book_path=p), embedder)


def test_blank_cover_alone_does_not_make_pdf_scanned(indexed):
    _, _, _, report = indexed
    assert 1 in report["pages"]["suspicious_pages"] or report["pages"]["total"] == 6
    assert report["status"] == "indexed"


def test_report_contents(indexed):
    settings, _, _, r = indexed
    assert r["pages"]["total"] == 6
    assert r["pages"]["printed_page_labels_available"] is True
    assert r["chapters"]["source"] == "outline"
    assert r["chapters"]["chapters"] == 4 and r["chapters"]["sections"] == 2
    assert "Table of Contents" in r["chapters"]["excluded_from_index"]
    assert r["chunks"]["count"] > 5
    assert r["chunks"]["max_tokens_seen"] <= r["chunks"]["model_max_tokens"]
    assert r["embedding"]["model"] == "fake/hashing-embedder" and r["embedding"]["dimension"] == 256
    assert any("NOT owner-reviewed" in w for w in r["warnings"])
    assert (settings.reports_dir / "ingestion_report.json").is_file()
    assert "# Ingestion report" in (settings.reports_dir / "ingestion_report.md").read_text()


def test_reingestion_is_idempotent_and_duplicate_free(indexed):
    settings, emb, store, first = indexed
    n = store.count()
    second = run_ingestion(settings, emb, store)
    assert second["status"] == "up_to_date"
    assert store.count() == n == first["chunks"]["count"]
    forced = run_ingestion(settings, emb, store, force=True)
    assert forced["status"] == "indexed" and store.count() == n  # rebuilt, still no duplicates
    ids = store._collection().get()["ids"]
    assert len(ids) == len(set(ids)) == n


def test_changed_config_builds_replacement_then_retires_old(indexed):
    settings, emb, store, first = indexed
    old = read_active_index(settings.active_index_path)
    s2 = replace(settings, chunk_size=250)
    r2 = run_ingestion(s2, emb, store)
    new = read_active_index(settings.active_index_path)
    assert r2["status"] == "indexed" and new.fingerprint != old.fingerprint
    assert old.collection in r2["retired_collections"]
    assert store.count() == r2["chunks"]["count"]


def test_failed_rebuild_keeps_working_index(indexed, monkeypatch):
    settings, emb, store, _ = indexed
    before = read_active_index(settings.active_index_path)
    count_before = store.count()

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "build_collection", boom)
    with pytest.raises(RuntimeError):
        run_ingestion(replace(settings, chunk_size=250), emb, store)
    after = read_active_index(settings.active_index_path)
    assert after.fingerprint == before.fingerprint  # pointer untouched
    assert store.count() == count_before  # old collection still serves queries


def test_changed_pdf_is_detected(indexed, tmp_path):
    settings, emb, store, first = indexed
    other = tmp_path / "other.pdf"
    c = canvas.Canvas(str(other))
    for i in range(3):
        c.drawString(72, 700, f"A different book, page {i}. " + "Words go here. " * 20)
        c.showPage()
    c.save()
    r = run_ingestion(replace(settings, book_path=other, chapters_path=tmp_path / "other_ch.json"), emb, store)
    assert r["status"] == "indexed" and r["fingerprint"] != first["fingerprint"]


def test_detect_chapters_only_writes_draft_and_skips_embedding(make_settings):
    class Boom(FakeEmbedder):
        def check_available(self):
            raise AssertionError("must not load the embedder")

    s = make_settings()
    r = run_ingestion(s, Boom(), detect_chapters_only=True)
    assert r["status"] == "chapters_drafted_only"
    data = json.loads(s.chapters_path.read_text())
    assert data["reviewed"] is False and data["entries"]


def test_strict_mode_requires_reviewed_mapping(make_settings, embedder):
    with pytest.raises(IngestionError, match="reviewed"):
        run_ingestion(make_settings(), embedder, strict=True)


def test_missing_embedding_model_is_reported_clearly():
    emb = SentenceTransformerEmbedder("definitely/not-a-real-model-xyz", local_only=True)
    with pytest.raises(EmbeddingUnavailableError, match="embedding model"):
        emb.check_available()
