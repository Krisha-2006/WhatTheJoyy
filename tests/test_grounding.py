import pytest

from config import FALLBACK_TEXT
from src.errors import GroundingFailure
from src.grounding import (
    lexical_support,
    normalize_for_match,
    quote_in_passage,
    validate_candidate,
    verify_with_llm,
)
from src.schemas import AnswerJudgement, CandidateAnswer, Claim, Evidence, RetrievedChunk
from tests.conftest import ScriptedModel

TEXT = ("Mirelle Odan became the keeper of Gannet Rock Lighthouse in the spring of 1887. "
        "She climbed the 214 steps of the tower every evening at dusk to light the great brass lamp.")


def ev(label="E1", text=TEXT):
    return Evidence(label, RetrievedChunk("c-" + label, 0, text, "ch", "Chapter", "", 2, 3, "1", 0, len(text), 0.9))


def cand(*claims, answerable=True, complete=True):
    return CandidateAnswer(answerable=answerable, complete=complete, claims=list(claims))


def fails(candidate, reason, evidence=None):
    with pytest.raises(GroundingFailure) as exc:
        validate_candidate(candidate, evidence or [ev()], 0.5)
    assert exc.value.reason.startswith(reason), exc.value.reason


def test_valid_candidate_passes():
    out = validate_candidate(
        cand(Claim(text="Mirelle climbed 214 steps every evening to light the lamp.", evidenceIds=["E1"],
                   quote="She climbed the 214 steps of the tower")), [ev()], 0.5)
    assert out[0].evidence_labels == ["E1"] and out[0].quote


def test_fabricated_evidence_id_is_rejected():
    fails(cand(Claim(text="Mirelle climbed 214 steps.", evidenceIds=["E7"])), "unknown_evidence_id:E7")
    fails(cand(Claim(text="Mirelle climbed 214 steps.", evidenceIds=["chunk-999"])), "unknown_evidence_id")


def test_claim_without_evidence_is_rejected():
    fails(cand(Claim(text="Mirelle climbed 214 steps.", evidenceIds=[])), "claim_without_evidence")


def test_invalid_quotation_is_rejected():
    fails(cand(Claim(text="Mirelle climbed 214 steps.", evidenceIds=["E1"],
                     quote="She climbed 300 steps of the tower")), "quote_not_in_cited_passage")


def test_quote_must_be_in_the_cited_passage_not_just_any_passage():
    other = ev("E2", "The boat named the Curlew arrived every second Thursday with flour.")
    fails(cand(Claim(text="The boat arrived every second Thursday.", evidenceIds=["E2"],
                     quote="She climbed the 214 steps")), "quote_not_in_cited_passage", [ev(), other])


def test_quote_with_ellipsis_is_rejected():
    assert not quote_in_passage("She climbed ... the tower", TEXT)


def test_quote_matching_ignores_whitespace_case_and_curly_quotes():
    assert quote_in_passage("SHE   climbed\nthe 214 steps", TEXT)
    assert normalize_for_match("It’s “fine” — really") == 'it\'s "fine" - really'


def test_unsupported_claim_is_rejected():
    fails(cand(Claim(text="Mirelle was born in Norway and trained as a sailor.", evidenceIds=["E1"])),
          "claim_lexically_unsupported")


def test_wrong_number_is_rejected():
    fails(cand(Claim(text="Mirelle climbed 300 steps of the tower every evening.", evidenceIds=["E1"])),
          "claim_lexically_unsupported")


def test_claim_cannot_borrow_support_from_an_uncited_passage():
    other = ev("E2", "Tobias Wren kept the readings in a green ledger.")
    fails(cand(Claim(text="Tobias Wren kept a green ledger.", evidenceIds=["E1"])),
          "claim_lexically_unsupported", [ev(), other])


def test_unanswerable_and_incomplete_models_fall_back():
    fails(cand(answerable=False), "model_declared_unanswerable")
    fails(cand(Claim(text="Mirelle climbed 214 steps.", evidenceIds=["E1"]), complete=False),
          "model_declared_incomplete")
    fails(cand(), "no_claims")


def test_model_returning_fallback_text_as_a_claim_is_rejected():
    fails(cand(Claim(text=FALLBACK_TEXT, evidenceIds=["E1"])), "model_returned_fallback_as_claim")


def test_lexical_support_ratio():
    ok, ratio = lexical_support("Mirelle climbed 214 steps", [TEXT], 0.5)
    assert ok and ratio == 1.0
    assert lexical_support("", [TEXT], 0.5) == (False, 0.0)


def _claims():
    return validate_candidate(
        cand(Claim(text="Mirelle climbed 214 steps every evening.", evidenceIds=["E1"])), [ev()], 0.5)


def test_per_claim_llm_verifier_rejection_and_partial_answers():
    claims = _claims()
    verify_with_llm(claims, [ev()], "q", ScriptedModel(), mode="per_claim")  # passes
    with pytest.raises(GroundingFailure, match="llm_verifier_rejected_claim"):
        verify_with_llm(claims, [ev()], "q", ScriptedModel(verify=False), mode="per_claim")
    with pytest.raises(GroundingFailure, match="partial"):
        verify_with_llm(claims, [ev()], "q", ScriptedModel(judge=AnswerJudgement(answers_question=True, fully_answers=False)), mode="per_claim")
    with pytest.raises(GroundingFailure, match="off_topic"):
        verify_with_llm(claims, [ev()], "q", ScriptedModel(judge=AnswerJudgement(answers_question=False, fully_answers=False)), mode="per_claim")


def test_single_call_review_mode():
    claims = _claims()
    m = ScriptedModel()
    verify_with_llm(claims, [ev()], "q", m, mode="single")
    assert m.review_calls == 1
    with pytest.raises(GroundingFailure, match="llm_review_unsupported_claim"):
        verify_with_llm(claims, [ev()], "q", ScriptedModel(verify=False), mode="single")
    with pytest.raises(GroundingFailure, match="llm_review_partial"):
        verify_with_llm(claims, [ev()], "q", ScriptedModel(judge=AnswerJudgement(answers_question=True, fully_answers=False)), mode="single")


def test_off_mode_makes_no_llm_calls():
    m = ScriptedModel(verify=False)
    verify_with_llm(_claims(), [ev()], "q", m, mode="off")
    assert getattr(m, "review_calls", 0) == 0


def test_supporting_sentence_is_verbatim_and_best_match():
    from src.grounding import supporting_sentence
    s = supporting_sentence("Mirelle climbed 214 steps of the tower", [TEXT])
    assert s.startswith("She climbed the 214 steps") and s in TEXT
    assert supporting_sentence("zebras gallop", [TEXT]) is None


def test_validated_claim_gets_a_deterministic_quote_when_model_gives_none():
    out = validate_candidate(cand(Claim(text="Mirelle climbed 214 steps every evening.", evidenceIds=["E1"])), [ev()], 0.5)
    assert out[0].quote and quote_in_passage(out[0].quote, TEXT)
