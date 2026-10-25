from config import FALLBACK_TEXT
from src.citations import build_answer, page_display, reference_from_evidence
from src.rag_pipeline import fallback_response
from src.schemas import Evidence, RetrievedChunk, ValidatedClaim


def ev(label, cid, page, printed, chapter="The Tide Tables", section=""):
    c = RetrievedChunk(cid, 0, f"text of {cid}", "c1", chapter, section, page - 1, page, printed, 0, 10, 0.9)
    return Evidence(label, c)


def test_page_display_never_invents_printed_label():
    assert page_display(20, None) == "PDF page 20"
    assert page_display(20, "8") == "PDF page 20 (printed page 8)"


def test_only_cited_chunks_become_references_in_order_of_use():
    evidence = [ev("E1", "a", 3, "1"), ev("E2", "b", 4, "2"), ev("E3", "c", 5, "3")]
    claims = [ValidatedClaim("Second fact", ["E3"], None), ValidatedClaim("First fact", ["E1", "E3"], "text of a")]
    answer, refs = build_answer(claims, evidence)
    assert [r.chunk_id for r in refs] == ["c", "a"]  # E2 was retrieved but never cited -> absent
    assert answer == "Second fact. [1] First fact. [2] [1]"
    assert refs[1].quote == "text of a" and refs[0].quote is None


def test_reference_carries_pdf_page_printed_label_and_chapter():
    r = reference_from_evidence(ev("E1", "a", 20, "8", section="Intro"))
    assert (r.pdf_page_number, r.printed_page_label, r.chapter, r.section) == (20, "8", "The Tide Tables", "Intro")
    r2 = reference_from_evidence(ev("E1", "a", 20, None, chapter=""))
    assert r2.printed_page_label is None and r2.chapter == "Chapter unavailable"
    assert r2.excerpt == "text of a"


def test_camel_case_serialisation_matches_api_contract():
    r = reference_from_evidence(ev("E1", "a", 20, None))
    dumped = r.model_dump(by_alias=True)
    assert {"chunkId", "chapter", "pdfPageNumber", "printedPageLabel", "excerpt"} <= dumped.keys()


def test_fallback_has_exact_text_and_empty_references():
    f = fallback_response()
    assert f.answer == FALLBACK_TEXT == "The selected book does not provide enough information to answer this question."
    assert f.status == "insufficient_evidence" and f.references == []
