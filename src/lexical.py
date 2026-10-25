"""Small in-memory BM25 keyword index over the book's chunks.

Why: dense embeddings (MiniLM) are weak on rare names/terms and exact phrases,
and a single similarity threshold is fragile. Keyword search catches those
cases; the two rankings are fused (Reciprocal Rank Fusion) in retrieval.py.
Pure Python, no extra dependency; a 600-chunk book searches in <5 ms.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from src.grounding import content_tokens

# Words that describe the *request*, not the book's subject.
QUESTION_BOILERPLATE = frozenset(
    """book author chapter page text passage according say says said explain explained describe
    describes discuss discusses mention mentions tell telling meant mean means define defined
    give summary summarize summarise overview list point points main idea ideas exactly really
    please kindly question answer""".split()
)


def query_terms(question: str) -> list[str]:
    terms = [t for t in content_tokens(question)]
    cleaned = [t for t in terms if t not in QUESTION_BOILERPLATE]
    return list(dict.fromkeys(cleaned or terms))  # de-duplicated, order kept


@dataclass
class LexicalHit:
    index: int  # position in the list the index was built from
    bm25: float
    coverage: float  # IDF-weighted fraction of query terms present in the doc


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [content_tokens(d) for d in documents]
        self.tf = [Counter(toks) for toks in self.doc_tokens]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.n = len(documents)
        self.avg_len = (sum(self.doc_len) / self.n) if self.n else 1.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.df = df
        self.max_idf = self._idf_from_df(1) if self.n else 1.0

    @staticmethod
    def _idf_from_df_n(df: int, n: int) -> float:
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _idf_from_df(self, df: int) -> float:
        return self._idf_from_df_n(df, self.n)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # Terms absent from the whole book get the MAX idf: if the question asks
        # about something the book never mentions, coverage must drop.
        return self._idf_from_df(df) if df else self.max_idf

    def search(self, question: str, n: int) -> list[LexicalHit]:
        terms = query_terms(question)
        if not terms or not self.n:
            return []
        weights = {t: self.idf(t) for t in terms}
        total = sum(weights.values()) or 1.0
        hits: list[LexicalHit] = []
        for i, tf in enumerate(self.tf):
            score, matched = 0.0, 0.0
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
            for t in terms:
                f = tf.get(t)
                if not f:
                    continue
                matched += weights[t]
                score += weights[t] * (f * (self.k1 + 1)) / (f + norm)
            if score > 0:
                hits.append(LexicalHit(i, score, matched / total))
        hits.sort(key=lambda h: h.bm25, reverse=True)
        return hits[:n]
