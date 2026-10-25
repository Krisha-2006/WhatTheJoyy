# BookWise Setup Guide (from scratch)

BookWise answers questions about **one fixed PDF book** and cites chapter and PDF page.
Everything runs on your own computer: Ollama (the language model), ChromaDB (search index),
FastAPI (backend) and Streamlit (web page).

> You will use **three terminal windows** at the end: Ollama, the API, and the UI.
> Commands are given for **macOS/Linux** and **Windows (PowerShell)**.

---

## 1. Install Python 3.11

* **macOS:** `brew install python@3.11` (or the installer from python.org)
* **Ubuntu/Debian:** `sudo apt install python3.11 python3.11-venv`
* **Windows:** download Python 3.11 from <https://www.python.org/downloads/> and tick **"Add python.exe to PATH"**.

Check: `python3.11 --version` (macOS/Linux) or `py -3.11 --version` (Windows).

## 2. Create a virtual environment

```bash
# macOS / Linux
cd BookWise
python3.11 -m venv .venv
source .venv/bin/activate
```
```powershell
# Windows PowerShell
cd BookWise
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
# If scripts are blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Your prompt now starts with `(.venv)`. Repeat the *activate* line in every new terminal.

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt     # includes pytest; use requirements.txt for runtime only
```
This downloads PyTorch (through `sentence-transformers`); expect 1–3 GB and several minutes.

## 4. Install Ollama

Download from <https://ollama.com/download> (macOS, Windows, Linux installers).
Linux one-liner: `curl -fsSL https://ollama.com/install.sh | sh`.
Verify: `ollama --version`. The Ollama app/service listens on `http://localhost:11434`.
If it is not running, start it in a terminal: `ollama serve`.

## 5. Download the language model

```bash
ollama pull llama3.2:3b
ollama list            # should show llama3.2:3b
```

## 6. Download / cache the embedding model

One-time download (~90 MB) into the Hugging Face cache; afterwards embedding runs offline:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('model cached')"
```
Optional: after this succeeds, set `EMBEDDING_LOCAL_ONLY=true` in `.env` to forbid network access.

## 7. Create `.env`

```bash
cp .env.example .env            # macOS / Linux
```
```powershell
Copy-Item .env.example .env     # Windows
```
Generate a secret and paste it as `SECRET_KEY=` in `.env`:
```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```
The API refuses to start without a real `SECRET_KEY`. Never commit `.env`.

## 8. Place your PDF

Copy your book to **`data/books/selected_book.pdf`** (rename it; the long original filename is fine
to keep elsewhere). The PDF must have selectable text. It is git-ignored because it is probably copyrighted.

## 9. Review the chapter mapping

```bash
python -m src.ingestion --detect-chapters-only
```
This reads the PDF bookmarks and writes a **draft** `data/config/chapters.json` (`"reviewed": false`).
A draft built from your supplied PDF is already included; regenerate it if you change books.

Open the file and check:

| Field | Meaning |
|---|---|
| `title` | Name shown in citations |
| `start_page` | **PDF viewer page** (1-based) where it begins |
| `level` | `0` = top-level bookmark. Entries with `level <= chapter_level` are *chapters*; deeper ones are *sections* shown after the chapter name |
| `anchor` | Heading text used to find a start **in the middle of a page** |
| `anchor_offset` | Alternative: exact character offset in the cleaned page text |
| `indexed` | `false` = excluded from search (title page, contents, copyright, ads…) |

Tips for your book: its bookmarks have *parts* at level 0 and *essays/questions* at level 1, so
the default (`chapter_level` 0) shows e.g. "IN PURSUIT › THE FUNDAMENTAL DUALITY". If you prefer sections as the
"chapter", set `"chapter_level": 1` in the file (or `CHAPTER_OUTLINE_DEPTH=1`).

When satisfied, change `"reviewed": false` to `"reviewed": true`. A reviewed file is **never overwritten**.
(An unreviewed file is regenerated on each run.) To demand review before indexing, use `--strict`.

## 10. Run ingestion

```bash
python -m src.ingestion
```
Ingestion is a separate, owner-only step (no web endpoint can trigger it). Stop the API first if it is running.
It validates the PDF, builds chunks, embeds them, builds a **new** index, verifies it, then switches to it.
Running it again with nothing changed prints `Status: up_to_date`.

## 11. Inspect the reports

Open `runtime/reports/ingestion_report.md` (and `.json`). Check:

* **Pages / suspicious pages:** blank covers are normal.
* **Chapters:** source (`outline`/`manual`/`heuristic`), entries, excluded items, anchor warnings.
* **Chunks:** count, max tokens vs the model limit, oversize splits.
* **Warnings:** especially "Chapter unavailable" chunks and "NOT owner-reviewed".

## 12. Start the API (terminal 1)

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```
Check <http://127.0.0.1:8000/health> → `{"status":"ok",...}` and <http://127.0.0.1:8000/docs>.
(Ollama must be running, from step 4.)

## 13. Start Streamlit (terminal 2)

```bash
streamlit run app.py
```
Open <http://localhost:8501>.

## 14. Create an account

*Sign up* tab → email + password (10+ chars, mixed character types) → then *Log in*.
Owners who want a private demo can set `ALLOW_SIGNUP=false` or `INVITE_CODE=...` in `.env` (restart the API).

## 15. Ask an answerable question

Pick a fact you know is in the book, e.g. a distinctive phrase from a chapter. You should get a short answer with
numbered sources like `[1] IN PURSUIT › … — PDF page 33`; expand a source to see the exact book text.
The first question is slower (models load into memory). Open the PDF at that **PDF page** (viewer page number) to verify.

## 16. Test an unanswerable question

Ask `What is the capital of France?` and then a plausible but absent detail. Expected result:

> The selected book does not provide enough information to answer this question.

with **no** sources. If an unrelated question returns an answer, raise `MIN_TOP_SIMILARITY` (see
[ARCHITECTURE.md](ARCHITECTURE.md#threshold-calibration)). Tick **Evidence-only mode** in the sidebar to see verbatim excerpts without
any generated explanation.

## 17. Run the tests

```bash
pytest                       # fast, no Ollama or downloads needed (uses a fake embedder and scripted model)
RUN_INTEGRATION=1 pytest -m integration       # macOS/Linux: real Ollama + real embeddings
```
```powershell
$env:RUN_INTEGRATION=1; pytest -m integration   # Windows
```

## 18. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `SECRET_KEY is missing…` when starting the API | Fill `SECRET_KEY` in `.env` (step 7). |
| `PDF not found at …` | File must be `data/books/selected_book.pdf`. |
| `only N of M pages contain extractable text` | PDF is scanned; OCR it first (e.g. `ocrmypdf`). |
| `Could not load embedding model` | Run step 6 with internet, or set `EMBEDDING_LOCAL_ONLY=false`. |
| UI shows "language model service (Ollama) is unavailable" | Start Ollama (`ollama serve`) and check `ollama list` shows the model; check `OLLAMA_BASE_URL`. |
| `index_not_ready` / "has not been indexed yet" | Run `python -m src.ingestion`, then restart the API. |
| Everything returns the fallback / wrong passages | Run `python -m src.diagnose "your question"`. It shows dense and keyword hits, and why each passage was kept or rejected. Lower `MIN_TOP_SIMILARITY` / `LEXICAL_TOP_COVERAGE` if good passages are rejected. See *Debugging retrieval and speed* below. |
| Answers are slow | Run `python -m src.diagnose --speed` (tokens/second) and `python -m src.diagnose "q" --ask` (time per stage). See *Debugging retrieval and speed* below. |
| After upgrading BookWise, answers/retrieval look unchanged | Re-run `python -m src.ingestion` (the index format changed; it rebuilds automatically). |
| `429 Too many requests` | Per-user limit (`RATE_LIMIT_PER_MINUTE`). |
| `database is locked` / Chroma errors during ingestion | Stop the API while ingesting; run only one ingestion at a time. |
| Chapter shows "Chapter unavailable" | Text appears before the first bookmark, or no outline: edit `chapters.json`. |
| Port already in use | Use `--port 8001` and set `API_BASE_URL=http://localhost:8001`. |

## Debugging retrieval and speed

**Retrieval** — see exactly what the app sees:
```bash
python -m src.diagnose "What is the difference between pleasure and bliss?"
```
You get three tables (best by dense similarity, best by keyword coverage, fused ranking), the PDF page and chapter of each passage, a `*` for passages that pass the policy, and the final DECISION with the reason.
* Good passage present but marked rejected → lower `MIN_TOP_SIMILARITY` (try 0.22) or `LEXICAL_TOP_COVERAGE` in `.env`.
* Good passage not in the list at all → the text may not have extracted well; open `runtime/reports/ingestion_report.md`, or try `CHUNK_SIZE=800` and re-ingest.
* Unrelated questions are being answered → raise the thresholds.
After writing ~20 questions in `data/config/eval_set.json`, `python -m src.calibrate` summarises how many answerable questions are accepted and how many unrelated ones rejected.

**Speed** — the time is almost all language-model work on your CPU/GPU:
```bash
python -m src.diagnose --speed                       # model load time and tokens/second
python -m src.diagnose "your question" --ask         # time per stage: retrieve_s, generate_s, verify_s
```
What helps, in order of impact: (1) a GPU or a faster machine; (2) `LLM_VERIFIER_MODE=off` (removes one LLM call; less safe) ; (3) `FINAL_K=3` and `MAX_CONTEXT_CHARS=3000` (less text to read); (4) Evidence-only mode in the sidebar (no LLM at all, instant); (5) keep `WARMUP_ON_START=true` and `OLLAMA_KEEP_ALIVE=30m` so the model is not reloaded. Close other heavy programs: if the model does not fit in free RAM it swaps and becomes 10× slower.
