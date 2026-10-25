from src.chapter_detection import build_segments, resolve_chapters
from src.chunking import ChunkStats, build_chunks
from src.pdf_reader import read_pdf
from tests.conftest import SYNTHETIC_PDF, FakeEmbedder


def _chunks(tmp_path, size=400, overlap=50, max_tokens=256, stats=None, min_chars=0):
    doc = read_pdf(SYNTHETIC_PDF)
    res = resolve_chapters(doc, tmp_path / "c.json", 0)
    segs, _ = build_segments(doc.pages, res.mapping)
    emb = FakeEmbedder(max_seq_length=max_tokens)
    chunks = build_chunks(
        segs, book_id="bk", source_filename="synthetic_book.pdf", chunk_size=size,
        chunk_overlap=overlap, count_tokens=emb.count_tokens, max_tokens=max_tokens, stats=stats,
        min_chunk_chars=min_chars,
    )
    return doc, chunks


def test_every_chunk_has_exact_page_provenance(tmp_path):
    doc, chunks = _chunks(tmp_path)
    assert chunks
    for c in chunks:
        page = doc.pages[c.pdf_page_index]
        assert c.pdf_page_number == page.number == c.pdf_page_index + 1
        assert page.text[c.start_char : c.end_char] == c.text  # exact offsets
        assert c.printed_page_label == page.label
        assert c.source_filename == "synthetic_book.pdf" and c.book_id == "bk"


def test_chunks_never_mix_chapters_or_pages(tmp_path):
    _, chunks = _chunks(tmp_path, size=2000)  # huge size: would merge if allowed
    for c in chunks:
        assert not ("rope and basket" in c.text and "first great storm" in c.text)
    p4 = [c for c in chunks if c.pdf_page_number == 4]
    assert {c.chapter_title for c in p4} == {"The Tide Tables", "Storm Signals"}
    assert len({c.pdf_page_number for c in chunks if "214 steps" in c.text}) == 1


def test_excluded_front_matter_is_not_indexed(tmp_path):
    _, chunks = _chunks(tmp_path)
    assert not any("Table of Contents" in c.text for c in chunks)
    # page 2 (ToC) is excluded; page 1 (cover text before the first bookmark) has no chapter
    assert {c.pdf_page_number for c in chunks} == {1, 3, 4, 5, 6}
    assert {c.chapter_title for c in chunks if c.pdf_page_number == 1} == {"Chapter unavailable"}


def test_ids_and_hashes_are_deterministic(tmp_path):
    _, a = _chunks(tmp_path)
    _, b = _chunks(tmp_path)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    assert all(c.text_hash and len(c.text_hash) == 40 for c in a)
    assert [c.chunk_index for c in a] == list(range(len(a)))


def test_oversized_chunks_are_split_not_truncated(tmp_path):
    stats = ChunkStats()
    _, small = _chunks(tmp_path, size=800, overlap=0, max_tokens=12, stats=stats)
    assert stats.oversize_splits > 0
    assert all(c.token_count <= 12 for c in small)
    # no text lost: every word of the source appears in some chunk
    doc = read_pdf(SYNTHETIC_PDF)
    words = set(doc.pages[2].text.split())
    got = set(" ".join(c.text for c in small if c.pdf_page_number == 3).split())
    assert words <= got


def test_overlap_stays_within_the_same_page_and_chapter(tmp_path):
    _, chunks = _chunks(tmp_path, size=200, overlap=60)
    for a, b in zip(chunks, chunks[1:]):
        if a.pdf_page_number != b.pdf_page_number:
            assert a.end_char <= len(read_pdf(SYNTHETIC_PDF).pages[a.pdf_page_index].text)
            assert b.start_char < 400  # new page starts fresh; no text carried over


def test_tiny_tail_fragments_are_merged_with_exact_provenance(tmp_path):
    _, plain = _chunks(tmp_path, size=250, overlap=40, min_chars=0)
    doc, merged = _chunks(tmp_path, size=250, overlap=40, min_chars=120)
    assert len(merged) < len(plain)
    assert sum(len(c.text) < 120 for c in merged) <= sum(len(c.text) < 120 for c in plain)
    for c in merged:  # merged text is still an exact slice of the page
        assert doc.pages[c.pdf_page_index].text[c.start_char : c.end_char] == c.text


def test_embedding_text_adds_chapter_context_but_stored_text_is_unchanged():
    from src.chunking import embedding_text
    assert embedding_text("Storm Signals", "Barometer", "body") == "Storm Signals - Barometer\nbody"
    assert embedding_text("Chapter unavailable", "", "body") == "body"


def test_token_limit_accounts_for_the_title_prefix(tmp_path):
    _, chunks = _chunks(tmp_path, size=800, overlap=0, max_tokens=30)
    from src.chunking import embedding_text
    for c in chunks:
        assert FakeEmbedder().count_tokens(embedding_text(c.chapter_title, c.section_title, c.text)) <= 30
