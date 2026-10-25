"""Chapter detection and page->chapter segmentation.

Interpretation of the priority rule
-----------------------------------
1. If data/config/chapters.json exists AND has "reviewed": true, it is the
   owner's final decision and is used verbatim ("manual").
2. Otherwise a draft is generated from the PDF bookmarks/outline ("outline").
3. If there is no usable outline, a conservative heading regex proposes
   chapters ("heuristic"). These are always flagged for owner review.
4. If nothing is found, text is labelled "Chapter unavailable".

Drafts are written to chapters.json with "reviewed": false so the owner can
edit them and flip the flag. A reviewed file is never overwritten.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from src.errors import ChapterMappingError
from src.pdf_reader import PdfDocument
from src.schemas import PageText, Segment

CHAPTER_UNAVAILABLE = "Chapter unavailable"

# Bookmarks matching these are front/back matter and are excluded from the
# index by default (the owner can set "indexed": true to override).
DEFAULT_EXCLUDE_RE = re.compile(
    r"^(title page|cover|half title|table of contents|contents|copyright( page)?|"
    r"also by .*|other books .*|about the author|about the publisher|dedication|"
    r"acknowledg(e)?ments?|index|.*meditation resort.*)$",
    re.IGNORECASE,
)

_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
)
_HEADING_RE = re.compile(
    rf"^(?P<kind>chapter|part)\s+(?P<num>\d{{1,3}}|[ivxlcdm]{{1,7}}|{_NUMBER_WORDS})\b\s*(?P<sep>[:.\-–—])?\s*(?P<title>.{{0,70}})$",
    re.IGNORECASE,
)


class EntryModel(BaseModel):
    id: str
    title: str
    start_page: int = Field(ge=1, description="One-based PDF page number")
    level: int = Field(default=0, ge=0)
    anchor: Optional[str] = Field(
        default=None, description="Heading text to locate the start inside the page"
    )
    anchor_offset: Optional[int] = Field(
        default=None, ge=0, description="Explicit character offset in the cleaned page text"
    )
    indexed: bool = True
    note: Optional[str] = None


class ChaptersFile(BaseModel):
    version: int = 1
    reviewed: bool = False
    source: Literal["outline", "heuristic", "manual", "none"] = "none"
    chapter_level: int = 0
    notes: list[str] = Field(default_factory=list)
    entries: list[EntryModel] = Field(default_factory=list)


@dataclass
class ChapterResolution:
    mapping: ChaptersFile
    warnings: list[str]
    anchor_misses: list[str]
    wrote_draft: bool


# ----------------------------------------------------------------- generation
def draft_from_outline(doc: PdfDocument, chapter_level: int) -> ChaptersFile:
    entries: list[EntryModel] = []
    for n, o in enumerate(doc.outline, start=1):
        excluded = bool(DEFAULT_EXCLUDE_RE.match(o.title))
        entries.append(
            EntryModel(
                id=f"e{n:03d}",
                title=o.title,
                start_page=o.page_number,
                level=o.depth,
                anchor=o.title,
                indexed=not excluded,
                note="Auto-excluded front/back matter" if excluded else None,
            )
        )
    return ChaptersFile(
        reviewed=False,
        source="outline",
        chapter_level=chapter_level,
        notes=[
            "Generated from the PDF bookmarks. Review titles, levels and 'indexed' flags, "
            'then set "reviewed": true. Entries at level <= chapter_level are chapters; '
            "deeper entries are sections."
        ],
        entries=entries,
    )


def draft_from_headings(doc: PdfDocument) -> ChaptersFile:
    entries: list[EntryModel] = []
    for page, lines in zip(doc.pages, doc.lines_by_page):
        non_empty = [ln for ln in lines if ln]
        for i, ln in enumerate(non_empty):
            if len(ln) > 90:
                continue
            m = _HEADING_RE.match(ln)
            if not m:
                continue
            title = (m.group("title") or "").strip()
            sep = m.group("sep")
            if title and not sep:
                continue  # "Chapter 3 explains how..." is prose, not a heading
            if not title and i + 1 < len(non_empty):
                nxt = non_empty[i + 1]
                if len(nxt) <= 70 and not nxt.endswith((",", ";")):
                    title = nxt
            full = f"{m.group('kind').title()} {m.group('num')}" + (f": {title}" if title else "")
            entries.append(
                EntryModel(
                    id=f"e{len(entries) + 1:03d}",
                    title=full,
                    start_page=page.number,
                    level=0,
                    anchor=ln,
                    note="Heuristic suggestion - verify",
                )
            )
    return ChaptersFile(
        reviewed=False,
        source="heuristic" if entries else "none",
        chapter_level=0,
        notes=["Generated by heading regex (no usable PDF outline). Every entry needs owner review."],
        entries=entries,
    )


# ------------------------------------------------------------------ file I/O
def load_chapters_file(path: Path, page_count: int) -> ChaptersFile:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        mapping = ChaptersFile.model_validate(data)
    except (json.JSONDecodeError, ValidationError, OSError) as exc:
        raise ChapterMappingError(f"{path} is not a valid chapter mapping file: {exc}") from exc
    for e in mapping.entries:
        if e.start_page > page_count:
            raise ChapterMappingError(
                f"Entry {e.id} ({e.title!r}) starts on PDF page {e.start_page}, "
                f"but the PDF has only {page_count} pages."
            )
    ids = [e.id for e in mapping.entries]
    if len(ids) != len(set(ids)):
        raise ChapterMappingError("Duplicate entry ids in chapter mapping file.")
    pages = [e.start_page for e in mapping.entries]
    if pages != sorted(pages):
        raise ChapterMappingError("Entries in the chapter mapping must be ordered by start_page.")
    return mapping


def write_chapters_file(path: Path, mapping: ChaptersFile) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(mapping.model_dump(), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def resolve_chapters(doc: PdfDocument, path: Path, chapter_level: int, write_draft: bool = True) -> ChapterResolution:
    warnings: list[str] = []
    path = Path(path)
    if path.is_file():
        existing = load_chapters_file(path, doc.page_count)
        if existing.reviewed:
            mapping = existing
            wrote = False
        else:
            mapping, wrote = _regenerate(doc, path, chapter_level, write_draft, warnings)
    else:
        mapping, wrote = _regenerate(doc, path, chapter_level, write_draft, warnings)

    if not mapping.reviewed:
        warnings.append(
            "Chapter mapping is NOT owner-reviewed (data/config/chapters.json has "
            '"reviewed": false). Citations may show imperfect chapter names until you review it.'
        )
    if not mapping.entries:
        warnings.append("No chapters detected; citations will say 'Chapter unavailable'.")
    return ChapterResolution(mapping=mapping, warnings=warnings, anchor_misses=[], wrote_draft=wrote)


def _regenerate(doc, path, chapter_level, write_draft, warnings):
    if doc.outline:
        mapping = draft_from_outline(doc, chapter_level)
    else:
        warnings.append("PDF has no usable bookmarks/outline; falling back to heading detection.")
        mapping = draft_from_headings(doc)
        mapping.chapter_level = chapter_level
    wrote = False
    if write_draft:
        write_chapters_file(path, mapping)
        wrote = True
    return mapping, wrote


# ----------------------------------------------------------------- segmentation
def _anchor_regex(text: str) -> re.Pattern:
    """Match on word tokens only, so quotes, dashes, symbols and line-wrap
    whitespace differences between bookmark titles and page text are ignored."""
    words = re.findall(r"\w+", text)
    if not words:
        raise re.error("empty anchor")
    return re.compile(r"\W+".join(re.escape(w) for w in words), re.IGNORECASE)


def locate_entry(entry: EntryModel, page: PageText) -> tuple[int, bool]:
    """Return (offset_in_page_text, found). Offset 0 if no anchor given/found."""
    if entry.anchor_offset is not None:
        return min(entry.anchor_offset, len(page.text)), True
    if not entry.anchor:
        return 0, True
    try:
        m = _anchor_regex(entry.anchor).search(page.text)
    except re.error:
        return 0, False
    if m:
        return m.start(), True
    return 0, False


@dataclass
class _Boundary:
    page_number: int
    offset: int
    entry: EntryModel


def build_segments(
    pages: list[PageText], mapping: ChaptersFile
) -> tuple[list[Segment], list[str]]:
    """Split every page at chapter/section boundaries (including mid-page)."""
    warnings: list[str] = []
    by_number = {p.number: p for p in pages}
    boundaries: list[_Boundary] = []
    for e in mapping.entries:
        page = by_number[e.start_page]
        off, found = locate_entry(e, page)
        if not found:
            warnings.append(
                f"Anchor for {e.title!r} was not found on PDF page {e.start_page}; "
                "it is assumed to start at the top of that page. Add 'anchor' or 'anchor_offset'."
            )
        boundaries.append(_Boundary(e.start_page, off, e))
    boundaries.sort(key=lambda b: (b.page_number, b.offset))

    # state while walking the boundaries in reading order
    def state_after(upto: int) -> tuple[Optional[EntryModel], Optional[EntryModel], bool]:
        chapter: Optional[EntryModel] = None
        section: Optional[EntryModel] = None
        for b in boundaries[:upto]:
            if b.entry.level <= mapping.chapter_level:
                chapter, section = b.entry, None
            else:
                section = b.entry
        indexed = True
        if chapter is not None and not chapter.indexed:
            indexed = False
        if section is not None and not section.indexed:
            indexed = False
        return chapter, section, indexed

    segments: list[Segment] = []
    for page in pages:
        if not page.text.strip():
            continue
        here = [(i, b) for i, b in enumerate(boundaries) if b.page_number == page.number]
        cuts: list[tuple[int, int]] = []  # (offset, boundary-count-before-this-text)
        first_idx = here[0][0] if here else None
        cuts.append((0, first_idx if first_idx is not None else _count_before(boundaries, page.number)))
        for i, b in here:
            cuts.append((b.offset, i + 1))
        cuts.sort(key=lambda c: (c[0], c[1]))
        # merge cuts at identical offsets keeping the last state
        merged: list[tuple[int, int]] = []
        for off, k in cuts:
            if merged and merged[-1][0] == off:
                merged[-1] = (off, k)
            else:
                merged.append((off, k))
        for n, (off, k) in enumerate(merged):
            end = merged[n + 1][0] if n + 1 < len(merged) else len(page.text)
            raw = page.text[off:end]
            lead = len(raw) - len(raw.lstrip())
            text = raw.strip()
            if not text:
                continue
            chapter, section, indexed = state_after(k)
            segments.append(
                Segment(
                    page_index=page.index,
                    page_number=page.number,
                    printed_label=page.label,
                    start=off + lead,
                    end=off + lead + len(text),
                    text=text,
                    chapter_id=chapter.id if chapter else None,
                    chapter_title=chapter.title if chapter else CHAPTER_UNAVAILABLE,
                    section_id=section.id if section else None,
                    section_title=section.title if section else None,
                    indexed=indexed,
                )
            )
    return segments, warnings


def _count_before(boundaries: list[_Boundary], page_number: int) -> int:
    return sum(1 for b in boundaries if b.page_number < page_number)
