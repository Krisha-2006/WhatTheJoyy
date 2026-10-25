import json
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from src.chapter_detection import (
    CHAPTER_UNAVAILABLE,
    ChaptersFile,
    EntryModel,
    build_segments,
    draft_from_headings,
    load_chapters_file,
    resolve_chapters,
)
from src.errors import ChapterMappingError
from src.pdf_reader import read_pdf
from tests.conftest import SYNTHETIC_PDF


@pytest.fixture
def doc():
    return read_pdf(SYNTHETIC_PDF)


def test_outline_draft_excludes_front_matter_and_keeps_levels(doc, tmp_path):
    res = resolve_chapters(doc, tmp_path / "chapters.json", chapter_level=0)
    titles = {e.title: e for e in res.mapping.entries}
    assert res.mapping.source == "outline" and res.mapping.reviewed is False
    assert titles["Table of Contents"].indexed is False
    assert titles["The Tide Tables"].level == 0
    assert titles["Reading the Barometer"].level == 1
    assert (tmp_path / "chapters.json").is_file()  # draft written for owner review


def test_chapter_starts_mid_page(doc, tmp_path):
    res = resolve_chapters(doc, tmp_path / "c.json", 0)
    segs, warnings = build_segments(doc.pages, res.mapping)
    page4 = [s for s in segs if s.page_number == 4]
    assert [s.chapter_title for s in page4] == ["The Tide Tables", "Storm Signals"]
    assert "rope and basket" in page4[0].text and "STORM SIGNALS" in page4[1].text
    assert "rope and basket" not in page4[1].text
    assert not warnings


def test_sections_inside_chapters(doc, tmp_path):
    res = resolve_chapters(doc, tmp_path / "c.json", 0)
    segs, _ = build_segments(doc.pages, res.mapping)
    p5 = [s for s in segs if s.page_number == 5][0]
    assert p5.chapter_title == "Storm Signals" and p5.section_title == "Reading the Barometer"
    p6 = [s for s in segs if s.page_number == 6]
    assert p6[0].section_title is None and p6[-1].section_title == "Winter Logbook"
    assert all(s.chapter_title == "The Quiet Season" for s in p6)


def test_pdf_page_versus_printed_label(doc):
    assert doc.has_reliable_page_labels
    p3 = doc.pages[2]
    assert (p3.index, p3.number, p3.label) == (2, 3, "1")  # PDF page 3 == printed page 1
    assert doc.pages[0].label == "i"


def test_printed_labels_are_not_invented_when_pdf_has_none(tmp_path):
    p = tmp_path / "nolabels.pdf"
    c = canvas.Canvas(str(p))
    for i in range(3):
        c.drawString(72, 700, "Some ordinary text on this page. " * 6)
        c.showPage()
    c.save()
    d = read_pdf(p)
    assert d.has_reliable_page_labels is False
    assert all(pg.label is None for pg in d.pages)


def test_reviewed_mapping_is_used_verbatim_and_never_overwritten(doc, tmp_path):
    path = tmp_path / "chapters.json"
    reviewed = ChaptersFile(
        reviewed=True, source="manual", chapter_level=0,
        entries=[EntryModel(id="m1", title="Owner Title", start_page=3, anchor=None)],
    )
    path.write_text(json.dumps(reviewed.model_dump()))
    before = path.read_text()
    res = resolve_chapters(doc, path, 0)
    assert [e.title for e in res.mapping.entries] == ["Owner Title"]
    assert path.read_text() == before and not res.wrote_draft
    segs, _ = build_segments(doc.pages, res.mapping)
    assert {s.chapter_title for s in segs if s.page_number >= 3} == {"Owner Title"}
    assert {s.chapter_title for s in segs if s.page_number < 3} == {CHAPTER_UNAVAILABLE}


def test_manual_anchor_offset_splits_page(doc):
    mapping = ChaptersFile(
        reviewed=True, source="manual",
        entries=[
            EntryModel(id="a", title="A", start_page=3),
            EntryModel(id="b", title="B", start_page=3, anchor_offset=200),
        ],
    )
    segs, _ = build_segments(doc.pages, mapping)
    p3 = [s for s in segs if s.page_number == 3]
    assert [s.chapter_title for s in p3] == ["A", "B"]
    assert p3[1].start >= 200


def test_unfound_anchor_warns_for_owner_review(doc):
    mapping = ChaptersFile(entries=[EntryModel(id="a", title="X", start_page=3, anchor="not on this page")])
    _, warnings = build_segments(doc.pages, mapping)
    assert warnings and "not found" in warnings[0]


def test_mapping_page_beyond_pdf_is_rejected(doc, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(ChaptersFile(entries=[EntryModel(id="a", title="T", start_page=99)]).model_dump()))
    with pytest.raises(ChapterMappingError):
        load_chapters_file(path, doc.page_count)


def test_heading_regex_fallback_is_conservative(tmp_path):
    p = tmp_path / "nooutline.pdf"
    c = canvas.Canvas(str(p))
    pages = [
        ["Chapter 1: Beginnings", "This is the opening text of the first chapter. " * 5],
        ["As explained, Chapter 3 explains the plan in detail and is not a heading at all.",
         "THIS LINE IS ALL UPPERCASE BUT IS NOT A CHAPTER", "Body text continues here. " * 8],
        ["CHAPTER II", "The Second Part", "More body text follows the heading here. " * 5],
    ]
    for lines in pages:
        y = 750
        for ln in lines:
            c.drawString(72, y, ln[:95]); y -= 18
        c.showPage()
    c.save()
    d = read_pdf(p)
    draft = draft_from_headings(d)
    titles = [e.title for e in draft.entries]
    assert titles == ["Chapter 1: Beginnings", "Chapter II: The Second Part"]
    res = resolve_chapters(d, tmp_path / "c.json", 0)
    assert res.mapping.source == "heuristic" and not res.mapping.reviewed
    assert any("NOT owner-reviewed" in w for w in res.warnings)


def test_no_detection_yields_chapter_unavailable(tmp_path):
    p = tmp_path / "plain.pdf"
    c = canvas.Canvas(str(p))
    for _ in range(2):
        c.drawString(72, 700, "Just ordinary prose without any headings whatsoever. " * 3)
        c.showPage()
    c.save()
    d = read_pdf(p)
    res = resolve_chapters(d, tmp_path / "c.json", 0)
    segs, _ = build_segments(d.pages, res.mapping)
    assert res.mapping.entries == [] and segs
    assert all(s.chapter_title == CHAPTER_UNAVAILABLE for s in segs)
