"""Shared fixtures and test doubles (fake embedder, scripted LLM)."""
from __future__ import annotations

import hashlib
import math
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from config import Settings
from src.errors import InvalidModelOutputError
from src.grounding import content_tokens
from src.ingestion import run_ingestion
from src.rag_pipeline import RagPipeline
from src.retrieval import RetrievalPolicy, Retriever
from src.schemas import AnswerJudgement, CandidateAnswer, Claim, Review
from src.vector_store import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC_PDF = FIXTURES / "synthetic_book.pdf"


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration test: set RUN_INTEGRATION=1 (needs Ollama + embedding model)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


class FakeEmbedder:
    """Deterministic bag-of-content-words hashing embedder (no downloads)."""

    model_name = "fake/hashing-embedder"

    def __init__(self, dim: int = 256, max_seq_length: int = 256):
        self._dim = dim
        self._max = max_seq_length
        self.query_calls = 0

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def max_seq_length(self) -> int:
        return self._max

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def check_available(self) -> None:
        return None

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        for tok in content_tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self._dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self._vec(text)

    def version_info(self) -> dict:
        return {
            "model": self.model_name,
            "dimension": self._dim,
            "max_seq_length": self._max,
            "sentence_transformers": "fake",
            "preprocess_version": "1",
            "normalized": True,
        }


class ScriptedModel:
    """Stands in for Ollama. `candidate` may be a CandidateAnswer, an exception
    instance to raise, or a callable(question, evidence) -> CandidateAnswer."""

    def __init__(self, candidate=None, verify=True, judge=None):
        self.candidate = candidate
        self.verify = verify
        self.judge = judge or AnswerJudgement(answers_question=True, fully_answers=True)
        self.generate_calls = 0
        self.last_evidence = None

    def generate(self, question, evidence):
        self.generate_calls += 1
        self.last_evidence = evidence
        c = self.candidate
        if isinstance(c, Exception):
            raise c
        if callable(c):
            return c(question, evidence)
        return c

    def verify_claim(self, claim, passages):
        return self.verify(claim, passages) if callable(self.verify) else self.verify

    def judge_answer(self, question, answer):
        return self.judge

    def review(self, question, items):
        self.review_calls = getattr(self, "review_calls", 0) + 1
        ok = all(self.verify(c, p) if callable(self.verify) else self.verify for c, p in items)
        return Review(
            all_supported=ok,
            unsupported_claims=[] if ok else [1],
            answers_question=self.judge.answers_question,
            fully_answers=self.judge.fully_answers,
        )


def first_sentence(text: str) -> str:
    return text.split(". ")[0].rstrip(".") + "."


def extractive(question, evidence):
    """A well-behaved fake model: answers from the sentences of the best-overlapping
    excerpt that share words with the question (max 3 claims)."""
    q = set(content_tokens(question))
    best = max(evidence, key=lambda e: len(q & set(content_tokens(e.chunk.text))))
    sentences = [s.strip() for s in best.chunk.text.replace("\n", " ").split(". ") if s.strip()]
    scored = [(len(q & set(content_tokens(s))), i, s) for i, s in enumerate(sentences)]
    top = sorted([t for t in scored if t[0] > 0], reverse=True)[:3]
    top.sort(key=lambda t: t[1])  # keep reading order
    claims = [Claim(text=s.rstrip(".") + ".", evidenceIds=[best.label], quote=s.rstrip(".")) for _, _, s in top]
    if not claims:
        return CandidateAnswer(answerable=False, complete=False, claims=[])
    return CandidateAnswer(answerable=True, complete=True, claims=claims)


@pytest.fixture
def make_settings(tmp_path):
    def _make(**overrides) -> Settings:
        base = Settings(
            book_path=SYNTHETIC_PDF,
            chapters_path=tmp_path / "config" / "chapters.json",
            chroma_dir=tmp_path / "chroma",
            runtime_dir=tmp_path / "runtime",
            secret_key="x" * 48,
            # thresholds suit the fake bag-of-words embedder, NOT the real model
            min_similarity=0.05,
            min_top_similarity=0.12,
            chunk_size=400,
            chunk_overlap=50,
        )
        return replace(base, **overrides)

    return _make


@pytest.fixture
def settings(make_settings):
    return make_settings()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def indexed(settings, embedder):
    """Ingest the synthetic book once per test into a temp ChromaDB."""
    store = VectorStore(settings.chroma_dir, settings.active_index_path)
    report = run_ingestion(settings, embedder, store)
    return settings, embedder, store, report


@pytest.fixture
def pipeline_factory(indexed):
    settings, embedder, store, _ = indexed

    def _make(model, **policy_overrides):
        policy = replace(RetrievalPolicy.from_settings(settings), **policy_overrides)
        return RagPipeline(settings, Retriever(policy, embedder, store), model)

    return _make
