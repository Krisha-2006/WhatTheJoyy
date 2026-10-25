"""Threshold calibration + evaluation:  python -m src.calibrate [--eval path] [--full]

Reads an evaluation set (questions, expected answerability, expected pages /
passage / key facts), embeds each question and reports the TOP cosine
SIMILARITY. Use it to choose MIN_TOP_SIMILARITY / MIN_SIMILARITY for YOUR book.

--full also runs the complete pipeline (needs Ollama) and checks answerability
and key facts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import ROOT, get_settings
from src.embeddings import get_embedder
from src.errors import BookWiseError
from src.retrieval import RetrievalPolicy, Retriever, filter_candidates
from src.vector_store import VectorStore

DEFAULT_EVAL = ROOT / "data" / "config" / "eval_set.json"


def load_eval(path: Path) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for it in items:
        if "question" not in it or "answerable" not in it:
            raise ValueError(f"Eval item needs 'question' and 'answerable': {it}")
    return items


def suggest_threshold(answerable_tops: list[float], unanswerable_tops: list[float]) -> dict:
    lo = min(answerable_tops) if answerable_tops else None
    hi = max(unanswerable_tops) if unanswerable_tops else None
    out = {"lowest_answerable_top": lo, "highest_unanswerable_top": hi}
    if lo is not None and hi is not None:
        if lo > hi:
            out["suggested_min_top_similarity"] = round((lo + hi) / 2, 3)
            out["note"] = "Classes separate cleanly on this eval set."
        else:
            out["suggested_min_top_similarity"] = round(lo - 0.01, 3)
            out["note"] = (
                "Classes OVERLAP: retrieval alone cannot separate them. Keep the threshold low "
                "enough not to lose answerable questions and rely on the grounding checks."
            )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(DEFAULT_EVAL))
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args(argv)
    settings = get_settings()
    try:
        items = load_eval(Path(args.eval))
        store = VectorStore(settings.chroma_dir, settings.active_index_path)
        embedder = get_embedder(settings.embedding_model, settings.embedding_local_only)
        policy = RetrievalPolicy.from_settings(settings)
        retriever = Retriever(policy, embedder, store)

        yes, no = [], []
        acc_ok = rej_ok = n_yes = n_no = 0
        print(f"{'id':<8}{'answerable':<12}{'dense':<8}{'kw':<6}{'gate':<9}{'hit@k':<7}question")
        for it in items:
            fused = retriever.candidates(it["question"])
            top = max((c.similarity or 0.0 for c in fused), default=0.0)
            kw = max((c.lex_coverage or 0.0 for c in fused), default=0.0)
            kept, reason = filter_candidates(fused, policy)
            (yes if it["answerable"] else no).append(top)
            hit = ""
            if it["answerable"]:
                n_yes += 1
                acc_ok += reason is None
                if it.get("expected_pages"):
                    hit = str(any(c.pdf_page_number in it["expected_pages"] for c in kept[: policy.final_k]))
            elif it.get("kind") != "plausible_absent":
                n_no += 1
                rej_ok += reason is not None
            print(f"{it.get('id', ''):<8}{str(it['answerable']):<12}{top:<8.3f}{kw:<6.2f}{'reject' if reason else 'accept':<9}{hit:<7}{it['question'][:60]}")
        print(f"\nGate: {acc_ok}/{n_yes} answerable accepted, {rej_ok}/{n_no} unrelated rejected "
              "(plausible-but-absent questions are expected to pass the gate; the model/grounding must decline them)")
        print("Dense-score suggestion:", json.dumps(suggest_threshold(yes, no), indent=2))

        if args.full:
            from src.generation import OllamaAnswerModel
            from src.rag_pipeline import RagPipeline

            pipe = RagPipeline(settings, retriever, OllamaAnswerModel(settings))
            correct = 0
            for it in items:
                out = pipe.ask_detailed(it["question"])
                answered = out.response.status == "answered"
                ok = answered == it["answerable"]
                if ok and answered and it.get("key_facts"):
                    text = out.response.answer.casefold()
                    ok = all(f.casefold() in text for f in it["key_facts"])
                correct += ok
                print(f"[{'PASS' if ok else 'FAIL'}] {it.get('id', '')} -> {out.reason}")
            print(f"\n{correct}/{len(items)} passed")
        return 0
    except (BookWiseError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {getattr(exc, 'detail', exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
