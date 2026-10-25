"""Layers 4-8 of the grounding pipeline.

All functions here are deterministic and model-free except
`verify_with_llm`, which calls an injected verifier. NONE of these checks can
mathematically guarantee zero hallucinations; see docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import re
import unicodedata

from config import FALLBACK_TEXT
from src.errors import GroundingFailure
from src.schemas import CandidateAnswer, Evidence, ValidatedClaim

MAX_CLAIMS = 6

_STOP = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being it its
    this that these those he she they them his her their you your we our i me my not no do does did have
    has had will would can could should may might shall about into than then so such there here what
    which who whom whose when where why how also very more most some any each other over under again
    only own same too just""".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_QUOTE_MAP = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\u00a0": " ", "′": "'"}
)


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP)
    return " ".join(text.split()).casefold()


def _stem(w: str) -> str:
    for suf in ("ing", "edly", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def content_tokens(text: str) -> list[str]:
    out = []
    for t in _TOKEN_RE.findall(normalize_for_match(text)):
        t = t.split("'")[0]
        if t.isdigit():
            out.append(t)
        elif len(t) >= 3 and t not in _STOP:
            out.append(_stem(t))
    return out


def lexical_support(claim: str, passages: list[str], min_overlap: float) -> tuple[bool, float]:
    """Cheap deterministic screen: do the claim's content words (and every
    number) occur in the cited passages? Paraphrases that use synonyms can
    fail this screen; failing safe (fallback) is intended."""
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return False, 0.0
    evidence_tokens = set()
    for p in passages:
        evidence_tokens.update(content_tokens(p))
    unique = set(claim_tokens)
    numbers = {t for t in unique if t.isdigit()}
    if not numbers <= evidence_tokens:
        return False, 0.0
    ratio = len(unique & evidence_tokens) / len(unique)
    return ratio >= min_overlap, ratio


def quote_in_passage(quote: str, passage: str) -> bool:
    q = normalize_for_match(quote).strip(" \"'")
    if len(q) < 5 or "..." in q or "…" in q:
        return False
    return q in normalize_for_match(passage)


_SENT_SPLIT = re.compile(r"(?<=[.!?”\"])\s+(?=[A-Z“\"‘'(])")


def supporting_sentence(claim: str, passages: list[str]) -> str | None:
    """Pick, deterministically, the verbatim sentence of the cited passages that
    best matches the claim. Used for the highlighted 'exact supporting text', so
    no quotation ever has to be (mis)copied by the language model."""
    claim_tokens = set(content_tokens(claim))
    sentences = [
        t.strip()
        for p in passages
        for t in _SENT_SPLIT.split(p.replace("\n", " "))
        if len(t.strip()) >= 15
    ]
    toks = [set(content_tokens(t)) for t in sentences]
    # rarer words (found in fewer sentences) are stronger evidence of "this is the sentence"
    weight = {w: 1.0 / sum(w in ts for ts in toks) for w in claim_tokens if any(w in ts for ts in toks)}
    best, best_score = None, 0.0
    for sent, ts in zip(sentences, toks):
        score = sum(weight.get(w, 0.0) for w in claim_tokens & ts)
        if score > best_score:
            best, best_score = sent, score
    return best


def validate_candidate(
    candidate: CandidateAnswer,
    evidence: list[Evidence],
    min_overlap: float,
) -> list[ValidatedClaim]:
    """Schema-level + citation + quote + lexical checks. Raises GroundingFailure."""
    if not candidate.answerable:
        raise GroundingFailure("model_declared_unanswerable")
    if not candidate.complete:
        raise GroundingFailure("model_declared_incomplete")
    if not candidate.claims:
        raise GroundingFailure("no_claims")
    if len(candidate.claims) > MAX_CLAIMS:
        raise GroundingFailure("too_many_claims")

    by_label = {e.label: e for e in evidence}
    validated: list[ValidatedClaim] = []
    for claim in candidate.claims:
        text = claim.text.strip()
        if not text:
            raise GroundingFailure("empty_claim")
        if normalize_for_match(text) == normalize_for_match(FALLBACK_TEXT):
            raise GroundingFailure("model_returned_fallback_as_claim")
        if not claim.evidenceIds:
            raise GroundingFailure("claim_without_evidence")
        labels: list[str] = []
        for raw in claim.evidenceIds:
            label = raw.strip()
            if label not in by_label:  # fabricated or out-of-context id
                raise GroundingFailure(f"unknown_evidence_id:{label}")
            if label not in labels:
                labels.append(label)
        passages = [by_label[l].chunk.text for l in labels]

        quote = (claim.quote or "").strip() or None
        if quote and not any(quote_in_passage(quote, p) for p in passages):
            raise GroundingFailure("quote_not_in_cited_passage")

        ok, ratio = lexical_support(text, passages, min_overlap)
        if not ok:
            raise GroundingFailure(f"claim_lexically_unsupported({ratio:.2f})")
        quote = quote or supporting_sentence(text, passages)
        validated.append(ValidatedClaim(text=text, evidence_labels=labels, quote=quote))
    return validated


def verify_with_llm(claims: list[ValidatedClaim], evidence: list[Evidence], question: str, model, mode: str = "single") -> None:
    """Layer 7b. Modes: "off" | "single" (one combined review call) | "per_claim"."""
    if mode == "off":
        return
    by_label = {e.label: e for e in evidence}
    items = [(c.text, [by_label[l].chunk.text for l in c.evidence_labels]) for c in claims]
    if mode == "single":
        review = model.review(question, items)
        if review.unsupported_claims or not review.all_supported:
            raise GroundingFailure("llm_review_unsupported_claim")
        if not review.answers_question:
            raise GroundingFailure("llm_review_off_topic")
        if not review.fully_answers:
            raise GroundingFailure("llm_review_partial")
        return
    for text, passages in items:
        if not model.verify_claim(text, passages):
            raise GroundingFailure("llm_verifier_rejected_claim")
    judgement = model.judge_answer(question, " ".join(c.text for c in claims))
    if not judgement.answers_question:
        raise GroundingFailure("llm_judge_answer_off_topic")
    if not judgement.fully_answers:
        raise GroundingFailure("llm_judge_answer_partial")
