# Architecture

## Application flowchart

```
OWNER SETUP — once, before users access the app
───────────────────────────────────────────────
Owner supplies data/books/selected_book.pdf
                 │
                 ▼
Extract text page by page (pypdf), keep PDF page numbers
                 │
                 ▼
Detect chapters (bookmarks → reviewed mapping → regex) → Owner reviews chapters.json
                 │
                 ▼
Page/chapter-aware chunks (LangChain splitter, token-checked)
                 │
                 ▼
Local SentenceTransformers embeddings (all-MiniLM-L6-v2, normalised)
                 │
                 ▼
New ChromaDB collection → verify → atomic switch (runtime/active_index.json)


USER QUESTION FLOW
──────────────────
Streamlit login ─▶ POST /ask (Bearer token) ─▶ auth + rate limit + length validation
                 │
                 ▼
Embed question ─▶ dense search (ChromaDB, cosine) + keyword search (BM25)  ─▶ fuse (RRF)
                 │
                 ▼
Relevance policy (dense similarity OR keyword coverage gates) ─ fail ─▶ FALLBACK
                 │ pass
                 ▼
Dedupe overlaps → (optional) neighbour chunks → bounded context (E1, E2 …)
                 │
                 ▼
Ollama llama3.2:3b, temperature 0, short structured JSON ─ invalid ─▶ FALLBACK
                 │
                 ▼
Validate: answerable/complete ▸ evidence ids exist ▸ lexical support ▸ code picks the
          verbatim supporting sentence ▸ ONE LLM review call (claims supported + answers question)
                 │ any failure ─▶ FALLBACK
                 ▼
Answer + [n] markers + references (chapter, PDF page, exact excerpt)

Operational failures (Ollama down, Chroma error, index missing, embedding model missing)
do NOT go to the fallback: they become HTTP 503/504 with a safe message.
```

## Directory structure

```
BookWise/
├── app.py                  Streamlit UI (thin client of the API; no upload/book management)
├── api.py                  FastAPI app factory: /auth/*, /ask, /book, /health
├── config.py               Settings from .env (all tunables), FALLBACK_TEXT
├── pytest.ini  requirements*.txt  .env.example  .gitignore  Dockerfile.api/.ui  docker-compose.yml
├── data/
│   ├── books/selected_book.pdf     (owner supplies; git-ignored)
│   └── config/chapters.json        (draft generated; owner reviews)
├── src/
│   ├── ingestion.py        Owner CLI: validate → chapters → chunk → embed → index → report
│   ├── pdf_reader.py       pypdf extraction, cleaning, validation, outline, page labels
│   ├── chapter_detection.py  chapters.json model, outline/regex drafts, mid-page segmentation
│   ├── chunking.py         page/chapter-aware chunking with token limit enforcement
│   ├── embeddings.py       SentenceTransformers wrapper (shared by chunks and questions)
│   ├── vector_store.py     ChromaDB wrapper, similarity conversion, versioned collections
│   ├── retrieval.py        hybrid (dense+keyword) fusion, policy gates, dedupe, context budget
│   ├── lexical.py          in-memory BM25 keyword index and question-term extraction
│   ├── prompts.py          system prompt + data-delimited templates
│   ├── generation.py       ChatOllama structured output, error classification
│   ├── grounding.py        claim/citation/quote/lexical validation + LLM verification
│   ├── citations.py        validated claims → answer text + references
│   ├── rag_pipeline.py     orchestration, fallback vs operational errors
│   ├── auth.py             SQLite accounts (Argon2), sessions, rate limiter
│   ├── schemas.py  errors.py
│   ├── diagnose.py         owner tool: explain retrieval decisions, time each stage, benchmark Ollama
│   └── calibrate.py        threshold calibration + evaluation runner
├── tests/ (+ fixtures/synthetic_book.pdf, eval_set.json)   docs/   chroma_db/   runtime/
```

`runtime/` holds `auth.sqlite3` (accounts only), `active_index.json` (which Chroma collection is live) and `reports/`.

## LangChain usage

LangChain provides `RecursiveCharacterTextSplitter` (`langchain-text-splitters`), `ChatPromptTemplate` (`langchain-core`)
and `ChatOllama` with Ollama structured output (`langchain-ollama`). ChromaDB is used through its own client (not the
`langchain-chroma` wrapper) because we need explicit control of cosine configuration, versioned collections and metadata queries.

## Citation metadata design

Each chunk stores: `chunk_id`, `chunk_index` (reading order), `book_id`, `source_filename`, `text_hash` (SHA-1),
`pdf_page_index` (0-based), `pdf_page_number` (1-based), `printed_page_label` (empty unless the PDF really defines labels),
`chapter_id`, `chapter_title`, `section_id`, `section_title`, `start_char`/`end_char` (exact offsets in the cleaned page text),
`token_count`, `embedding_model`, `embedding_dimension`, `index_fingerprint`.

The LLM never sees page numbers or chunk ids. It sees short labels (`E1…E5`) and may only cite those. The backend maps labels to
metadata **after** validation, so a model cannot invent a page or chapter. Only evidence cited by surviving claims appears in `references`.
A quote is attached only to the passage that actually contains it.

Display: **"PDF page 20"** (viewer number) and, only when reliable, **"printed page 8"**.

## The grounding pipeline and its honest limits

| # | Layer | Catches | Cannot catch |
|---|---|---|---|
| 1 | Retrieval from the fixed book only | outside-topic questions | questions that merely *sound* similar |
| 2 | Relevance policy (similarity thresholds) | weak/irrelevant evidence | relevant-looking but non-answering passages |
| 3 | Short structured generation with evidence IDs (the model never copies quotes) | free-form rambling | – |
| 4 | Pydantic schema validation | malformed output | – |
| 5 | Evidence-ID check | fabricated/out-of-context ids | wrong-but-existing id (partly caught by 7) |
| 6 | Supporting text: the highlighted quote is chosen **by code** as the best-matching verbatim sentence of the cited passage (a model-supplied quote, if any, must be a verbatim substring) | invented quotations | the sentence may support only part of a claim |
| 7a | Lexical support (content words and **all numbers** must occur in cited passages; ≥50 %) | invented names, numbers, topics | negation flips, relation swaps, synonyms (may *wrongly reject* good paraphrases, which fails safe) |
| 7b | ONE LLM review call (`LLM_VERIFIER_MODE=single`): each claim vs its own passages, and does the answer address the whole question. `per_claim` = one call per claim (slower); `off` = skip | many paraphrase errors, off-topic/partial answers | errors by the same small model; the judge can be fooled |
| 8 | Fallback on any failure | – | – |

**None of this mathematically guarantees zero hallucination.** Specifically: a 3B model can misread a passage in a way
that still passes word overlap and its own judge; instructions *inside the book* are retrievable text and can be quoted
faithfully (they are fenced as data and cannot grant citations to unsupported claims, but a faithful claim "the note says X" is still a claim);
`temperature=0` reduces randomness but does not make output correct; "complete" answers spanning multiple passages are the weakest area.
The design bias is to **fail safe** (fallback) — expect some false fallbacks. Use evidence-only mode when certainty matters, and
always verify citations against the PDF page.

Why not rely on the system prompt alone? Because prompts are requests, not constraints. Here the code enforces what the prompt asks for.

## Hybrid retrieval (dense + keyword)

Dense embeddings (MiniLM) capture meaning but are weak on rare terms, names and exact phrases, and a single similarity threshold is fragile.
BookWise therefore runs two searches and fuses them with Reciprocal Rank Fusion:

* **Dense:** ChromaDB cosine search (top 30 by default).
* **Keyword:** an in-memory BM25 index over the same chunks (`src/lexical.py`). Request words like "book", "explain", "according to" are ignored. For each chunk it also computes **coverage**: the IDF-weighted share of the question's terms the chunk contains. Terms the book never uses count against coverage, which is what rejects "What is the capital of France?" without needing a tuned dense score.

Relevance policy: the question is rejected unless best dense similarity ≥ `MIN_TOP_SIMILARITY` **or** best keyword coverage ≥ `LEXICAL_TOP_COVERAGE`; a chunk is kept if dense ≥ `MIN_SIMILARITY` **or** coverage ≥ `LEXICAL_MIN_COVERAGE`. Kept chunks go in fused order; at most `FINAL_K` (default 4) reach the model.
Chunks are embedded **with their chapter/section title prepended** (stored and cited text stay unchanged), and tiny page-tail fragments are merged into the previous chunk, so passages carry context.
Use `python -m src.diagnose "question"` to see every number behind a decision.

## Similarity metric

Collections use **cosine** space. Chroma returns cosine *distance* `d = 1 − cos(a, b)`. `VectorStore.query` converts once:
`similarity = 1 − d` (1 = identical direction, ~0 = unrelated). All thresholds, logs and docs use **similarity**. Vectors are L2-normalised
at embedding time for both book and questions.

## Threshold calibration

The defaults (`MIN_TOP_SIMILARITY=0.28`, `MIN_SIMILARITY=0.20`, keyword coverage 0.6 / 0.5) are **starting guesses for all-MiniLM-L6-v2**,
not universal values (v1 shipped stricter 0.40/0.30 dense-only values, which most likely rejected valid questions).

1. Write `data/config/eval_set.json` for *your* book (format in TESTING_GUIDE): ≥15 answerable questions with the page you know holds the answer, and ≥15 unanswerable (unrelated and plausible-but-absent).
2. `python -m src.calibrate` prints each question's best similarity and whether the expected page was among the top-K.
3. Use `python -m src.diagnose` on the failures. Set `MIN_TOP_SIMILARITY` between the highest *unanswerable* and lowest *answerable* score. If they overlap, prefer a lower threshold (don't lose real answers) and rely on layers 4–8.
4. `python -m src.calibrate --full` (needs Ollama) runs the whole pipeline and checks answerability + key facts.
5. Re-calibrate after changing the book, chunk size, or embedding model.

## Authentication and security summary

Argon2id password hashes; random 256-bit opaque session tokens stored only as HMAC-SHA256 under `SECRET_KEY`; expiry + logout deletion;
generic login errors with timing equalisation; per-user `/ask` limit, per-IP+email login limit, hourly signup limit (in-memory, per process);
`Cache-Control: no-store` on answers; no question/answer text in logs; `extra="forbid"` on `/ask` so clients cannot pass book ids, context or filters;
user questions and book text are defanged (`<`/`>`) before entering delimited prompt blocks; HTTPS must be provided by a reverse proxy.
