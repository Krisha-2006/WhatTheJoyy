"""Page-by-page PDF extraction with pypdf, plus text cleaning and validation."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.errors import NoExtractableTextError, PdfMissingError, PdfUnreadableError
from src.schemas import PageText

# A page counts as "content" if it has at least this many characters.
CONTENT_PAGE_MIN_CHARS = 100
# A page is reported as "suspicious" below this many characters.
SUSPICIOUS_PAGE_MAX_CHARS = 50
# The PDF is rejected only if fewer than this share of pages are content pages.
MIN_CONTENT_PAGE_RATIO = 0.20


@dataclass
class OutlineEntry:
    title: str
    page_number: int  # one-based
    depth: int


@dataclass
class PdfDocument:
    path: Path
    sha256: str
    title: Optional[str]
    author: Optional[str]
    pages: list[PageText]
    lines_by_page: list[list[str]]  # cleaned, un-reflowed lines (for heading detection)
    outline: list[OutlineEntry]
    has_reliable_page_labels: bool
    warnings: list[str] = field(default_factory=list)
    suspicious_pages: list[int] = field(default_factory=list)  # one-based
    removed_boilerplate: dict[str, int] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)


# --------------------------------------------------------------------- helpers
_WS_RE = re.compile(r"[ \t\u00a0\u2009\u202f]+")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_line(line: str) -> str:
    return _WS_RE.sub(" ", line).strip()


def reflow_lines(lines: list[str]) -> str:
    """Rebuild paragraphs from hard-wrapped PDF lines.

    A line break is treated as a paragraph break when the line is clearly
    shorter than the page's longest line AND ends with terminal punctuation
    (this also isolates short headings). Everything else is joined with a
    space. This is a heuristic; the original PDF has no paragraph markup.
    """
    non_empty = [ln for ln in lines if ln]
    if not non_empty:
        return ""
    max_len = max(len(ln) for ln in non_empty)
    paragraphs: list[list[str]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            paragraphs.append(current)
            current = []

    for ln in lines:
        if not ln:
            flush()
            continue
        current.append(ln)
        if len(ln) < 0.75 * max_len and ln[-1] in ".?!\"”’)…:":
            flush()
    flush()
    return "\n\n".join(" ".join(p) for p in paragraphs)


def _margin_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.casefold())


_PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?(?:\d{1,4}|[ivxlc]{1,7})$", re.IGNORECASE)


def strip_boilerplate(lines_by_page: list[list[str]]) -> tuple[list[list[str]], dict[str, int]]:
    """Remove running headers/footers and bare page numbers, conservatively.

    A first/last line is removed only if it is (a) a bare page number, or
    (b) short and repeated (digits ignored) on at least 30% of text pages and
    at least 5 pages. Everything removed is counted for the ingestion report.
    """
    text_pages = [ln for ln in lines_by_page if any(ln)]
    removed: Counter = Counter()
    if not text_pages:
        return lines_by_page, {}

    def margins(lines: list[str]) -> list[tuple[int, str]]:
        idx = [i for i, ln in enumerate(lines) if ln]
        picks = idx[:2] + idx[-2:]
        return [(i, lines[i]) for i in dict.fromkeys(picks)]

    counts: Counter = Counter()
    for lines in text_pages:
        for _, ln in margins(lines):
            counts[_margin_key(ln)] += 1
    threshold = max(5, int(0.3 * len(text_pages)))
    repeated = {k for k, c in counts.items() if c >= threshold and len(k) <= 80}

    cleaned: list[list[str]] = []
    for lines in lines_by_page:
        drop: set[int] = set()
        for i, ln in margins(lines):
            if _PAGE_NUMBER_RE.match(ln):
                drop.add(i)
                removed["<page number>"] += 1
            elif _margin_key(ln) in repeated:
                drop.add(i)
                removed[ln[:60]] += 1
        cleaned.append([ln for i, ln in enumerate(lines) if i not in drop])
    return cleaned, dict(removed)


def _walk_outline(reader: PdfReader, items, depth: int, out: list[OutlineEntry], warnings: list[str]) -> None:
    for item in items:
        if isinstance(item, list):
            _walk_outline(reader, item, depth + 1, out, warnings)
            continue
        try:
            title = normalize_line(str(item.title))
            page = reader.get_destination_page_number(item) + 1
        except Exception:  # broken bookmark
            warnings.append("A bookmark with an unresolvable destination was ignored.")
            continue
        if not title or page < 1 or page > len(reader.pages):
            warnings.append(f"Bookmark {title!r} points to an invalid page and was ignored.")
            continue
        out.append(OutlineEntry(title=title, page_number=page, depth=depth))


def _reliable_labels(reader: PdfReader) -> list[Optional[str]]:
    """Return printed page labels only if the PDF really defines /PageLabels.

    pypdf synthesises "1, 2, 3..." labels when the PDF has none; those are NOT
    printed page numbers, so they are discarded.
    """
    try:
        root = reader.trailer["/Root"]
        if "/PageLabels" not in root:
            return [None] * len(reader.pages)
        labels = list(reader.page_labels)
        default = [str(i + 1) for i in range(len(labels))]
        if labels == default:
            return [None] * len(labels)
        return [str(x) for x in labels]
    except Exception:
        return [None] * len(reader.pages)


# ------------------------------------------------------------------ main entry
def read_pdf(path: Path) -> PdfDocument:
    path = Path(path)
    if not path.is_file():
        raise PdfMissingError(
            f"PDF not found at {path}. Place your book at data/books/selected_book.pdf."
        )
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            if not reader.decrypt(""):
                raise PdfUnreadableError("The PDF is password-protected and cannot be read.")
        n_pages = len(reader.pages)
    except PdfUnreadableError:
        raise
    except (PdfReadError, Exception) as exc:  # noqa: BLE001 - pypdf raises many types
        raise PdfUnreadableError(f"The PDF could not be opened/parsed: {exc}") from exc
    if n_pages == 0:
        raise PdfUnreadableError("The PDF has no pages.")

    warnings: list[str] = []
    raw_lines_by_page: list[list[str]] = []
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"PDF page {i + 1}: text extraction failed ({exc}).")
            raw = ""
        raw_lines_by_page.append([normalize_line(ln) for ln in raw.split("\n")])

    lines_by_page, removed = strip_boilerplate(raw_lines_by_page)
    labels = _reliable_labels(reader)

    pages: list[PageText] = []
    for i, lines in enumerate(lines_by_page):
        text = reflow_lines(lines)
        pages.append(PageText(index=i, number=i + 1, label=labels[i], text=text, char_count=len(text)))

    content_pages = [p for p in pages if p.char_count >= CONTENT_PAGE_MIN_CHARS]
    suspicious = [p.number for p in pages if p.char_count < SUSPICIOUS_PAGE_MAX_CHARS]
    ratio = len(content_pages) / n_pages
    if ratio < MIN_CONTENT_PAGE_RATIO:
        raise NoExtractableTextError(
            f"Only {len(content_pages)} of {n_pages} pages contain extractable text "
            f"({ratio:.0%}). The PDF looks scanned/image-only; OCR it first. "
            "(A few blank pages such as covers are normal and are not a problem.)"
        )
    if suspicious:
        warnings.append(
            f"{len(suspicious)} page(s) have almost no text (covers, blanks or images): {suspicious[:30]}"
            + (" ..." if len(suspicious) > 30 else "")
        )
    total_chars = sum(p.char_count for p in pages) or 1
    bad = sum(p.text.count("\ufffd") for p in pages)
    if bad / total_chars > 0.01:
        warnings.append("More than 1% of characters are U+FFFD replacement characters; the font encoding may be broken.")

    outline: list[OutlineEntry] = []
    try:
        _walk_outline(reader, reader.outline, 0, outline, warnings)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"PDF outline could not be read ({exc}).")

    meta = reader.metadata
    title = normalize_line(str(meta.title)) if meta and meta.title else None
    author = normalize_line(str(meta.author)) if meta and meta.author else None
    if not title:
        warnings.append("PDF metadata has no title; set BOOK_TITLE in .env to display one.")

    return PdfDocument(
        path=path,
        sha256=sha256_file(path),
        title=title,
        author=author,
        pages=pages,
        lines_by_page=lines_by_page,
        outline=outline,
        has_reliable_page_labels=any(x is not None for x in labels),
        warnings=warnings,
        suspicious_pages=suspicious,
        removed_boilerplate=removed,
    )
