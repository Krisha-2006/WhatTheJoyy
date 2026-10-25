"""Owner debugging tool.

    python -m src.diagnose "What is the difference between pleasure and bliss?"
    python -m src.diagnose "..." --ask        # also run the full pipeline (needs Ollama), with stage timings
    python -m src.diagnose --speed            # measure Ollama load time and tokens/second

It prints what the dense search and the keyword search each found, the fused
ranking, and WHY each passage was kept or rejected by the relevance policy, so
"retrieval is bad" becomes a concrete number you can tune in .env.
"""
from __future__ import annotations

import argparse
import sys
import time

from config import Settings, get_settings
from src.errors import BookWiseError
from src.retrieval import Retriever, RetrievalPolicy, filter_candidates


def _row(c, marker: str = " ") -> str:
    sim = "  -  " if c.similarity is None else f"{c.similarity:5.3f}"
    cov = "  -  " if c.lex_coverage is None else f"{c.lex_coverage:4.2f}"
    chap = (c.chapter_title or "")[:24]
    return f"{marker} dense={sim} kw={cov}  PDF p.{c.pdf_page_number:<4} [{chap}] {c.text[:80]!r}"


def diagnose(question: str, retriever: Retriever, out=print, top: int = 8) -> dict:
    policy: RetrievalPolicy = retriever.policy
    t = time.perf_counter()
    fused = retriever.candidates(question)
    total = time.perf_counter() - t
    kept, reason = filter_candidates(fused, policy)
    kept_ids = {c.chunk_id for c in kept}
    out(f"\nQuestion: {question}")
    out(f"Policy : dense top >= {policy.min_top_similarity} or keyword top >= {policy.lexical_top_coverage}; "
        f"chunk kept if dense >= {policy.min_similarity} or keyword >= {policy.lexical_min_coverage}; hybrid={policy.hybrid_enabled}")
    out(f"Timing : {retriever.last_timings}  (total {total:.2f}s)")
    out("\nBest by DENSE similarity:")
    for c in sorted([c for c in fused if c.similarity is not None], key=lambda c: -c.similarity)[:top]:
        out(_row(c, "*" if c.chunk_id in kept_ids else " "))
    if policy.hybrid_enabled:
        out("\nBest by KEYWORD coverage:")
        for c in sorted([c for c in fused if c.lex_coverage is not None], key=lambda c: -c.lex_coverage)[:top]:
            out(_row(c, "*" if c.chunk_id in kept_ids else " "))
    out("\nFused ranking (* = passes policy):")
    for c in fused[:top]:
        out(_row(c, "*" if c.chunk_id in kept_ids else " "))
    if reason:
        out(f"\nDECISION: REJECTED -> fallback.  Reason: {reason}")
        out("  If this question IS answered in the book, lower MIN_TOP_SIMILARITY / LEXICAL_TOP_COVERAGE in .env.")
    else:
        out(f"\nDECISION: ACCEPTED ({len(kept)} passages pass; top {policy.final_k} go to the model)")
    return {"fused": fused, "kept": kept, "reason": reason}


def ollama_speed(settings: Settings, out=print) -> None:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0,
                     num_ctx=settings.ollama_num_ctx, num_predict=60, keep_alive=settings.ollama_keep_alive,
                     client_kwargs={"timeout": settings.ollama_timeout_seconds})
    for label in ("1st call (may include model load)", "2nd call (warm)"):
        t = time.perf_counter()
        msg = llm.invoke("Count from one to twenty in words, separated by commas.")
        wall = time.perf_counter() - t
        m = msg.response_metadata
        load = m.get("load_duration", 0) / 1e9
        n = m.get("eval_count", 0)
        ev = (m.get("eval_duration", 1) or 1) / 1e9
        out(f"{label}: wall {wall:.1f}s, load {load:.1f}s, generated {n} tokens at {n / ev:.1f} tok/s")
    out("\nGuide: a full answer needs ~100-250 generated tokens plus reading the ~1000-token context. "
        "Below ~8 tok/s expect 20-40 s per answer on this model; consider evidence-only mode, "
        "LLM_VERIFIER_MODE=off, FINAL_K=3, or a GPU / smaller context.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--ask", action="store_true", help="also run the full pipeline (needs Ollama)")
    ap.add_argument("--speed", action="store_true", help="benchmark Ollama")
    args = ap.parse_args(argv)
    settings = get_settings()
    try:
        if args.speed:
            ollama_speed(settings)
        if args.question:
            from api import build_default_pipeline
            from src.vector_store import VectorStore

            store = VectorStore(settings.chroma_dir, settings.active_index_path)
            pipe = build_default_pipeline(settings, store)
            diagnose(args.question, pipe.retriever)
            if args.ask:
                out = pipe.ask_detailed(args.question)
                print(f"\nANSWER ({out.response.status}, reason={out.reason}):\n{out.response.answer}")
                for i, r in enumerate(out.response.references, 1):
                    print(f"  [{i}] {r.chapter} - PDF page {r.pdf_page_number}: {r.quote or r.excerpt[:100]!r}")
                print("\nStage timings:", {k: v for k, v in out.trace.items() if k.endswith("_s")})
        if not args.question and not args.speed:
            ap.print_help()
        return 0
    except BookWiseError as exc:
        print(f"ERROR [{exc.code}]: {exc.detail}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
