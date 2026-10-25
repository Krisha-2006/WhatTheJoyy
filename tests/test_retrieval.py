import json
from dataclasses import replace

import pytest

from src.errors import IndexNotReadyError, VectorStoreError
from src.lexical import BM25Index, query_terms
from src.retrieval import RetrievalPolicy, Retriever, assemble_context, dedupe, filter_candidates, fuse
from src.schemas import RetrievedChunk
from src.vector_store import VectorStore
from tests.conftest import FIXTURES


def _rc(i, sim, text="x", page=1, start=0, end=100, chapter="c1"):
    return RetrievedChunk(f"id{i}", i, text, chapter, "T", "", page - 1, page, None, start, end, sim)


def test_relevant_question_retrieves_expected_page(indexed):
    settings, emb, store, _ = indexed
    r = Retriever(RetrievalPolicy.from_settings(settings), emb, store).retrieve(
        "How many steps did Mirelle climb to light the lamp?"
    )
    assert any("214 steps" in e.chunk.text for e in r.evidence)
    assert [e.label for e in r.evidence] == [f"E{i}" for i in range(1, len(r.evidence) + 1)]


def test_unrelated_question_is_rejected_by_policy(indexed):
    settings, emb, store, _ = indexed
    r = Retriever(RetrievalPolicy.from_settings(settings), emb, store).retrieve("What is the capital of France?")
    assert r.evidence == [] and r.rejected_reason


def test_weak_retrieval_is_rejected():
    policy = RetrievalPolicy(min_top_similarity=0.5, min_similarity=0.3, lexical_top_coverage=0.6)
    kept, reason = filter_candidates([_rc(1, 0.2), _rc(2, 0.1)], policy)
    assert kept == [] and "weak_evidence" in reason


def test_policy_accepts_on_strong_keywords_even_if_dense_is_weak():
    from dataclasses import replace
    policy = RetrievalPolicy(min_top_similarity=0.5, min_similarity=0.3)
    a = replace(_rc(1, 0.1), lex_coverage=0.9)
    b = replace(_rc(2, 0.1), lex_coverage=0.2)
    kept, reason = filter_candidates([a, b], policy)
    assert reason is None and [c.chunk_id for c in kept] == ["id1"]


def test_policy_per_chunk_floor_keeps_fused_order():
    policy = RetrievalPolicy(min_top_similarity=0.4, min_similarity=0.3)
    kept, _ = filter_candidates([_rc(1, 0.8), _rc(2, 0.75), _rc(3, 0.5), _rc(4, 0.1)], policy)
    assert [c.chunk_id for c in kept] == ["id1", "id2", "id3"]


def test_rrf_fusion_prefers_chunks_found_by_both_searches():
    dense = [_rc(1, 0.6), _rc(2, 0.5), _rc(3, 0.4)]
    lexical = [(_rc(3, None), 0.9), (_rc(9, None), 0.8)]
    fused = fuse(dense, lexical)
    assert fused[0].chunk_id == "id3" and fused[0].lex_coverage == 0.9 and fused[0].similarity == 0.4
    assert {c.chunk_id for c in fused} == {"id1", "id2", "id3", "id9"}
    lex_only = next(c for c in fused if c.chunk_id == "id9")
    assert lex_only.similarity is None and lex_only.lex_coverage == 0.8


def test_keyword_search_finds_rare_terms_dense_would_miss():
    idx = BM25Index(["the lamp burned whale oil", "ginger biscuits in a tin", "tide tables in a blue notebook"])
    hits = idx.search("Where are the biscuits?", 3)
    assert hits[0].index == 1 and hits[0].coverage > 0.6


def test_terms_absent_from_the_book_lower_keyword_coverage():
    idx = BM25Index(["the lamp burned whale oil", "ginger biscuits in a tin"])
    assert idx.search("capital of France lamp", 3)[0].coverage < 0.5  # 'capital' and 'france' unseen


def test_request_boilerplate_words_are_ignored_in_keyword_search():
    assert query_terms("According to the book, explain the lighthouse lamp") == ["lighthous", "lamp"] or \
        set(query_terms("According to the book, explain the lighthouse lamp")) >= {"lamp"}
    assert "book" not in query_terms("what does the book say about lamp")


def test_similarity_is_one_minus_cosine_distance(indexed):
    settings, emb, store, _ = indexed
    chunk_text = store._collection().get(limit=1, include=["documents"])["documents"][0]
    top = store.query(emb.embed_query(chunk_text), 1)[0]
    assert top.similarity == pytest.approx(1.0, abs=1e-3)  # identical text -> similarity 1


def test_dedupe_removes_overlapping_and_identical_chunks():
    a = _rc(1, 0.9, "same text", page=3, start=0, end=100)
    b = _rc(2, 0.8, "other text", page=3, start=40, end=140)  # >50% overlap on same page
    c = _rc(3, 0.7, "Same  Text", page=5, start=0, end=100)  # identical text, other page
    d = _rc(4, 0.6, "distinct", page=3, start=300, end=400)
    assert [x.chunk_id for x in dedupe([a, b, c, d])] == ["id1", "id4"]


def test_context_budget_never_truncates_or_drops_best_chunk():
    chunks = [_rc(i, 0.9 - i / 10, "y" * 500) for i in range(5)]
    ctx = assemble_context(chunks, [], max_chars=1200)
    assert len(ctx) == 2 and all(len(c.text) == 500 for c in ctx)
    assert assemble_context(chunks[:1], [], max_chars=10)[0].chunk_id == "id0"


def test_neighbor_expansion_keeps_each_chunks_own_page_and_same_chapter(indexed):
    settings, emb, store, _ = indexed
    policy = replace(RetrievalPolicy.from_settings(settings), neighbor_expansion=1, final_k=1)
    r = Retriever(policy, emb, store).retrieve("What was the supply boat called?")
    neighbors = [e for e in r.evidence if e.chunk.is_neighbor]
    assert neighbors
    primary = next(e for e in r.evidence if not e.chunk.is_neighbor)
    for n in neighbors:
        assert n.chunk.chapter_id == primary.chunk.chapter_id
        assert n.chunk.pdf_page_number >= 1 and n.chunk.chunk_id != primary.chunk.chunk_id


def test_no_active_index_is_operational_error(settings):
    store = VectorStore(settings.chroma_dir, settings.active_index_path)
    with pytest.raises(IndexNotReadyError):
        store.query([0.0] * 4, 1)


def test_chromadb_failure_is_operational_not_insufficient_evidence(indexed, monkeypatch):
    settings, emb, store, _ = indexed
    col = store._collection()

    class Broken:
        def query(self, **kw):
            raise RuntimeError("segment corrupted")

    monkeypatch.setattr(store, "_collection", lambda: Broken())
    with pytest.raises(VectorStoreError):
        store.query([0.1] * 256, 3)


def test_eval_set_retrieval_hits_expected_pages(indexed):
    settings, emb, store, _ = indexed
    r = Retriever(RetrievalPolicy.from_settings(settings), emb, store)
    for item in json.loads((FIXTURES / "eval_set.json").read_text()):
        res = r.retrieve(item["question"])
        if item["kind"] == "answerable":
            assert any(item["expected_passage_contains"] in e.chunk.text for e in res.evidence), item["id"]
            hit = next(e for e in res.evidence if item["expected_passage_contains"] in e.chunk.text)
            assert hit.chunk.pdf_page_number in item["expected_pages"], item["id"]
            assert hit.chunk.chapter_title == item["expected_chapter"], item["id"]
        elif item["kind"] == "unrelated":
            assert res.evidence == [], item["id"]


def test_diagnose_explains_acceptance_and_rejection(indexed):
    from src.diagnose import diagnose
    settings, emb, store, _ = indexed
    r = Retriever(RetrievalPolicy.from_settings(settings), emb, store)
    lines = []
    ok = diagnose("What was the supply boat called?", r, out=lines.append)
    assert ok["reason"] is None and any("DECISION: ACCEPTED" in l for l in lines)
    assert any("PDF p.3" in l and "*" in l[:2] for l in lines)
    lines.clear()
    bad = diagnose("What is the capital of France?", r, out=lines.append)
    assert bad["reason"] and any("REJECTED" in l for l in lines)


def test_calibrate_runs_on_fixture_set(indexed, capsys, monkeypatch):
    import src.calibrate as cal
    settings, emb, store, _ = indexed
    monkeypatch.setattr(cal, "get_settings", lambda: settings)
    monkeypatch.setattr(cal, "get_embedder", lambda *a, **k: emb)
    monkeypatch.setattr(cal, "VectorStore", lambda *a, **k: store)
    assert cal.main(["--eval", str(FIXTURES / "eval_set.json")]) == 0
    out = capsys.readouterr().out
    assert "6/6 answerable accepted" in out and "2/2 unrelated rejected" in out
