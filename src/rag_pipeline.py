"""The full question -> answer pipeline."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from config import FALLBACK_TEXT, Settings
from src.citations import build_answer, reference_from_evidence
from src.errors import GroundingFailure, InvalidModelOutputError
from src.generation import AnswerModel
from src.grounding import validate_candidate, verify_with_llm
from src.retrieval import Retriever
from src.schemas import AskResponse

log = logging.getLogger("bookwise.rag")

EVIDENCE_ONLY_TEXT = (
    "Relevant excerpts from the book are shown below. No explanation was generated; "
    "these passages were selected by similarity search, not interpreted by the language model."
)


@dataclass
class PipelineOutcome:
    response: AskResponse
    reason: str  # machine-readable; logged and tested, never shown to users
    trace: dict = field(default_factory=dict)


def fallback_response() -> AskResponse:
    return AskResponse(status="insufficient_evidence", answer=FALLBACK_TEXT, references=[])


class RagPipeline:
    def __init__(self, settings: Settings, retriever: Retriever, model: AnswerModel):
        self.settings = settings
        self.retriever = retriever
        self.model = model

    def warmup(self) -> None:
        """Pre-load embedding model, keyword index and the Ollama model. Never raises."""
        for name, fn in (("retriever", self.retriever.warmup), ("llm", getattr(self.model, "warmup", None))):
            if fn is None:
                continue
            try:
                t = time.perf_counter()
                fn()
                log.info("warm-up %s done in %.1fs", name, time.perf_counter() - t)
            except Exception as exc:  # noqa: BLE001 - warm-up must never block startup
                log.warning("warm-up %s skipped: %s", name, exc)

    def ask(self, question: str, evidence_only: bool = False) -> AskResponse:
        return self.ask_detailed(question, evidence_only).response

    def ask_detailed(self, question: str, evidence_only: bool = False) -> PipelineOutcome:
        t0 = time.perf_counter()
        trace: dict = {"question_chars": len(question), "evidence_only": evidence_only}

        # Operational errors (VectorStoreError, EmbeddingUnavailableError,
        # LLMUnavailableError...) deliberately propagate to the API layer.
        t_ret = time.perf_counter()
        retrieval = self.retriever.retrieve(question)
        trace["retrieve_s"] = round(time.perf_counter() - t_ret, 2)
        trace.update(getattr(self.retriever, "last_timings", {}))
        trace.update(
            top_similarity=retrieval.top_similarity,
            candidates=retrieval.candidates_considered,
            evidence=len(retrieval.evidence),
        )
        if not retrieval.evidence:
            return self._done(fallback_response(), f"retrieval:{retrieval.rejected_reason}", trace, t0)

        if evidence_only:
            primary = [e for e in retrieval.evidence if not e.chunk.is_neighbor]
            refs = [reference_from_evidence(e) for e in primary]
            resp = AskResponse(status="answered", answer=EVIDENCE_ONLY_TEXT, references=refs)
            return self._done(resp, "evidence_only", trace, t0)

        try:
            t_gen = time.perf_counter()
            candidate = self.model.generate(question, retrieval.evidence)
            trace["generate_s"] = round(time.perf_counter() - t_gen, 2)
        except InvalidModelOutputError:
            return self._done(fallback_response(), "invalid_model_output", trace, t0)

        try:
            claims = validate_candidate(
                candidate, retrieval.evidence, self.settings.grounding_min_lexical_overlap
            )
            t_ver = time.perf_counter()
            verify_with_llm(claims, retrieval.evidence, question, self.model, self.settings.llm_verifier_mode)
            trace["verify_s"] = round(time.perf_counter() - t_ver, 2)
        except GroundingFailure as failure:
            return self._done(fallback_response(), f"grounding:{failure.reason}", trace, t0)

        answer, refs = build_answer(claims, retrieval.evidence)
        resp = AskResponse(status="answered", answer=answer, references=refs)
        return self._done(resp, "answered", trace, t0)

    @staticmethod
    def _done(resp: AskResponse, reason: str, trace: dict, t0: float) -> PipelineOutcome:
        trace["elapsed_s"] = round(time.perf_counter() - t0, 2)
        trace["reason"] = reason
        resp.elapsed_seconds = trace["elapsed_s"]
        # Private content (question/answer text) is intentionally NOT logged.
        log.info("ask status=%s reason=%s trace=%s", resp.status, reason, trace)
        return PipelineOutcome(resp, reason, trace)
