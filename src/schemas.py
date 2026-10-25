"""Pydantic models (API + LLM output) and plain dataclasses (pipeline objects)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from pydantic.alias_generators import to_camel


# ---------------------------------------------------------------- API models
class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)
    invite_code: Optional[str] = Field(default=None, max_length=128, alias="inviteCode")
    model_config = ConfigDict(populate_by_name=True)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(CamelModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: str


class MessageResponse(BaseModel):
    message: str


class AskRequest(CamelModel):
    """Only the question (and an optional display-mode flag) is accepted.

    There is deliberately no book id, no context field and no retrieval filter.
    Extra fields are rejected.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    question: str = Field(max_length=5000)  # hard cap; the configurable limit is enforced in the API
    evidence_only: bool = False

    @field_validator("question")
    @classmethod
    def _clean_question(cls, v: str) -> str:
        v = " ".join(v.split())  # collapse whitespace/newlines
        if not v:
            raise ValueError("Question must not be empty.")
        return v


class Reference(CamelModel):
    chunk_id: str
    chapter: str
    section: Optional[str] = None
    pdf_page_number: int
    printed_page_label: Optional[str] = None
    excerpt: str
    quote: Optional[str] = None


class AskResponse(CamelModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    references: list[Reference] = Field(default_factory=list)
    elapsed_seconds: Optional[float] = None


class BookInfo(CamelModel):
    title: str
    author: Optional[str] = None
    page_count: int
    chunk_count: int
    embedding_model: str
    indexed_at: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


# ----------------------------------------------------- LLM structured output
class Claim(BaseModel):
    text: str
    evidenceIds: list[str] = Field(default_factory=list)
    quote: Optional[str] = None


class CandidateAnswer(BaseModel):
    answerable: bool
    complete: bool = True
    claims: list[Claim] = Field(default_factory=list)


class Verdict(BaseModel):
    supported: bool
    reason: str = ""


# ------------------------------------------------------ pipeline dataclasses
@dataclass
class PageText:
    index: int  # zero-based internal index
    number: int  # one-based PDF viewer page number
    label: Optional[str]  # printed page label, only if reliably available
    text: str  # cleaned, reflowed text
    char_count: int = 0


@dataclass
class Segment:
    """A piece of one page that belongs to exactly one chapter/section."""

    page_index: int
    page_number: int
    printed_label: Optional[str]
    start: int
    end: int
    text: str
    chapter_id: Optional[str]
    chapter_title: str
    section_id: Optional[str]
    section_title: Optional[str]
    indexed: bool = True


@dataclass
class Chunk:
    chunk_id: str
    chunk_index: int
    book_id: str
    source_filename: str
    text: str
    text_hash: str
    pdf_page_index: int
    pdf_page_number: int
    printed_page_label: Optional[str]
    chapter_id: str
    chapter_title: str
    section_id: str
    section_title: str
    start_char: int
    end_char: int
    token_count: int


@dataclass
class RetrievedChunk:
    chunk_id: str
    chunk_index: int
    text: str
    chapter_id: str
    chapter_title: str
    section_title: str
    pdf_page_index: int
    pdf_page_number: int
    printed_page_label: Optional[str]
    start_char: int
    end_char: int
    similarity: Optional[float]  # dense cosine similarity; None if not in dense top-N
    is_neighbor: bool = False
    lex_coverage: Optional[float] = None  # IDF-weighted share of question terms found (0..1)
    score: float = 0.0  # fused (RRF) rank score


@dataclass
class Evidence:
    label: str  # short id shown to the LLM: E1, E2, ...
    chunk: RetrievedChunk


@dataclass
class RetrievalResult:
    evidence: list[Evidence]
    top_similarity: Optional[float]
    candidates_considered: int
    rejected_reason: Optional[str] = None


@dataclass
class ValidatedClaim:
    text: str
    evidence_labels: list[str]
    quote: Optional[str]


class AnswerJudgement(BaseModel):
    answers_question: bool
    fully_answers: bool = True
    reason: str = ""


class Review(BaseModel):
    """One combined LLM review of a whole candidate answer (fast verifier)."""

    all_supported: bool
    answers_question: bool = True
    fully_answers: bool = True
    unsupported_claims: list[int] = Field(default_factory=list)
    reason: str = ""
