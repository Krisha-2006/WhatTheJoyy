"""Ollama access through LangChain (ChatOllama), with structured JSON output.

Outage vs. bad output
---------------------
* Ollama unreachable / model not pulled / timeout -> LLMUnavailableError /
  LLMTimeoutError (operational, HTTP 503/504).
* Ollama answered but the JSON is invalid        -> InvalidModelOutputError
  (the pipeline treats that as a failed validation => fallback).
"""
from __future__ import annotations

import json
import logging
from typing import Protocol

import httpx
from pydantic import BaseModel, ValidationError

from config import Settings
from src.errors import InvalidModelOutputError, LLMTimeoutError, LLMUnavailableError
from src.prompts import (
    ANSWER_PROMPT,
    JUDGE_PROMPT,
    REVIEW_PROMPT,
    format_review_items,
    VERIFY_PROMPT,
    format_excerpts,
    format_passages,
    sanitize_question,
)
from src.schemas import AnswerJudgement, CandidateAnswer, Evidence, Review, Verdict

log = logging.getLogger("bookwise.generation")


class AnswerModel(Protocol):
    def generate(self, question: str, evidence: list[Evidence]) -> CandidateAnswer: ...
    def verify_claim(self, claim: str, passages: list[str]) -> bool: ...
    def judge_answer(self, question: str, answer: str) -> AnswerJudgement: ...
    def review(self, question: str, items: list[tuple[str, list[str]]]) -> Review: ...


def classify_ollama_error(exc: Exception, model: str, base_url: str) -> Exception:
    if isinstance(exc, httpx.TimeoutException):
        return LLMTimeoutError(f"Ollama timed out: {exc}")
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, ConnectionError, OSError)):
        return LLMUnavailableError(f"Cannot reach Ollama at {base_url}: {exc}")
    name = type(exc).__name__
    if name == "ResponseError":
        status = getattr(exc, "status_code", None)
        if status == 404:
            return LLMUnavailableError(f"Ollama model {model!r} is not installed. Run: ollama pull {model}")
        return LLMUnavailableError(f"Ollama returned an error ({status}): {exc}")
    return LLMUnavailableError(f"Unexpected error talking to Ollama: {name}: {exc}")


class OllamaAnswerModel:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._chat_cache: dict[str, object] = {}

    # ------------------------------------------------------------------ llm
    def _chat(self, schema_model: type[BaseModel]):
        key = schema_model.__name__
        if key not in self._chat_cache:
            from langchain_ollama import ChatOllama

            s = self.settings
            schema = schema_model.model_json_schema()
            if schema_model is CandidateAnswer:  # the model must not copy quotes; we pick them deterministically
                schema.get("$defs", {}).get("Claim", {}).get("properties", {}).pop("quote", None)
            self._chat_cache[key] = ChatOllama(
                model=s.ollama_model,
                base_url=s.ollama_base_url,
                temperature=0,  # reduces randomness; does NOT guarantee correctness
                seed=0,
                num_ctx=s.ollama_num_ctx,
                num_predict=s.ollama_num_predict if schema_model is CandidateAnswer else 160,
                keep_alive=s.ollama_keep_alive,  # keep the model in memory between questions
                format=schema,  # Ollama structured output
                client_kwargs={"timeout": s.ollama_timeout_seconds},
            )
        return self._chat_cache[key]

    def _run(self, prompt, schema_model: type[BaseModel], variables: dict, attempts: int = 2):
        chain = prompt | self._chat(schema_model)
        last: Exception | None = None
        for _ in range(attempts):
            try:
                message = chain.invoke(variables)
            except Exception as exc:  # noqa: BLE001
                raise classify_ollama_error(exc, self.settings.ollama_model, self.settings.ollama_base_url) from exc
            content = message.content
            if isinstance(content, list):
                content = "".join(part if isinstance(part, str) else part.get("text", "") for part in content)
            try:
                return schema_model.model_validate(json.loads(content))
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                last = exc
                log.warning("Model returned invalid %s JSON (attempt failed).", schema_model.__name__)
        raise InvalidModelOutputError(f"Model output did not match {schema_model.__name__}: {last}")

    # ------------------------------------------------------------- public
    def generate(self, question: str, evidence: list[Evidence]) -> CandidateAnswer:
        return self._run(
            ANSWER_PROMPT,
            CandidateAnswer,
            {"excerpts": format_excerpts(evidence), "question": sanitize_question(question)},
        )

    def verify_claim(self, claim: str, passages: list[str]) -> bool:
        try:
            v: Verdict = self._run(
                VERIFY_PROMPT, Verdict, {"passages": format_passages(passages), "claim": sanitize_question(claim)}
            )
        except InvalidModelOutputError:
            return False  # an unreadable verdict is never treated as approval
        return bool(v.supported)

    def judge_answer(self, question: str, answer: str) -> AnswerJudgement:
        try:
            return self._run(
                JUDGE_PROMPT,
    REVIEW_PROMPT,
    format_review_items,
                AnswerJudgement,
                {"question": sanitize_question(question), "answer": sanitize_question(answer)},
            )
        except InvalidModelOutputError:
            return AnswerJudgement(answers_question=False, fully_answers=False, reason="unreadable verdict")

    def review(self, question: str, items: list[tuple[str, list[str]]]) -> Review:
        try:
            return self._run(
                REVIEW_PROMPT,
                Review,
                {"items": format_review_items(items), "question": sanitize_question(question)},
            )
        except InvalidModelOutputError:
            return Review(all_supported=False, answers_question=False, fully_answers=False, reason="unreadable verdict")

    def warmup(self) -> None:
        """Load the model into memory now so the first user question is not slow."""
        from langchain_ollama import ChatOllama

        s = self.settings
        ChatOllama(
            model=s.ollama_model, base_url=s.ollama_base_url, num_predict=1, num_ctx=s.ollama_num_ctx,
            keep_alive=s.ollama_keep_alive, client_kwargs={"timeout": s.ollama_timeout_seconds},
        ).invoke("ok")

    # ------------------------------------------------------------- health
    def status(self) -> dict:
        s = self.settings
        try:
            r = httpx.get(f"{s.ollama_base_url.rstrip('/')}/api/tags", timeout=2.0)
            r.raise_for_status()
            names = {m.get("name", "") for m in r.json().get("models", [])}
            wanted = s.ollama_model
            present = wanted in names or any(n.split(":")[0] == wanted and ":" not in wanted for n in names)
            return {"reachable": True, "model_installed": present}
        except Exception:  # noqa: BLE001
            return {"reachable": False, "model_installed": False}
