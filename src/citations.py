"""Turn validated claims into the user-facing answer and chapter/page citations."""
from __future__ import annotations

from typing import Optional

from src.chapter_detection import CHAPTER_UNAVAILABLE
from src.grounding import quote_in_passage
from src.schemas import Evidence, Reference, ValidatedClaim


def page_display(pdf_page: int, printed_label: Optional[str]) -> str:
    """'PDF page 20' or 'PDF page 20 (printed page 8)'. Never invents a label."""
    if printed_label:
        return f"PDF page {pdf_page} (printed page {printed_label})"
    return f"PDF page {pdf_page}"


def reference_from_evidence(ev: Evidence, quote: Optional[str] = None) -> Reference:
    c = ev.chunk
    return Reference(
        chunk_id=c.chunk_id,
        chapter=c.chapter_title or CHAPTER_UNAVAILABLE,
        section=c.section_title or None,
        pdf_page_number=c.pdf_page_number,
        printed_page_label=c.printed_page_label,
        excerpt=c.text,
        quote=quote,
    )


def build_answer(claims: list[ValidatedClaim], evidence: list[Evidence]) -> tuple[str, list[Reference]]:
    """Only chunks actually cited by validated claims become references."""
    by_label = {e.label: e for e in evidence}
    order: list[str] = []
    quotes: dict[str, Optional[str]] = {}
    for c in claims:
        for label in c.evidence_labels:
            if label not in order:
                order.append(label)
        if c.quote:
            # attach the quote only to the passage that really contains it
            owner = next((l for l in c.evidence_labels if quote_in_passage(c.quote, by_label[l].chunk.text)), None)
            if owner and not quotes.get(owner):
                quotes[owner] = c.quote
    number = {label: i for i, label in enumerate(order, start=1)}

    sentences = []
    for c in claims:
        text = c.text.rstrip()
        if text[-1] not in ".!?\"”":
            text += "."
        sentences.append(text + "".join(f" [{number[l]}]" for l in c.evidence_labels))
    refs = [reference_from_evidence(by_label[l], quotes.get(l)) for l in order]
    return " ".join(sentences), refs
