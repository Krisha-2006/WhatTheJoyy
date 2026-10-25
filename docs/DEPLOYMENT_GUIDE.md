# Deployment Guide

## Why this exact stack is not an unchanged Vercel deployment

You mentioned Vercel earlier. BookWise's required stack needs **long-running processes and local disk**:

| Needs | Vercel (serverless/edge functions) |
|---|---|
| A running **Ollama** server holding a multi-GB model in RAM | Not possible: functions are short-lived, memory/size-limited, no GPU/daemon |
| **Persistent local ChromaDB** + SQLite files | Function filesystems are ephemeral/read-only; data would vanish between invocations |
| A long-lived **Streamlit** server (websocket sessions) | Not a supported runtime for Streamlit |
| Local **PyTorch/SentenceTransformers** inference | Exceeds bundle/memory/time limits |
| In-process rate limiting and model warm state | Instances are stateless and recycled |

You *could* host only a static front end on Vercel and run the rest elsewhere, but that is a different architecture (and Streamlit would be replaced), which you asked me not to do.
Therefore: **run it locally or on a self-hosted machine/VM/container host with persistent disks and enough RAM.**

## Free software vs. potentially paid things

| Free / open-source software | Possible costs (not guaranteed free anywhere) |
|---|---|
| Python, FastAPI, Uvicorn, Streamlit, LangChain, ChromaDB, SQLite, pypdf, pytest | A **server/VM** with ≥8 GB RAM and persistent disk (monthly fee on most providers) |
| Ollama and the `llama3.2:3b` model (check the model's licence terms for your use) | A **GPU** machine if you want fast answers or larger models |
| SentenceTransformers + all-MiniLM-L6-v2 (download once) | A **domain name**, TLS (Let's Encrypt itself is free), backups/storage, bandwidth |
| Docker / Docker Compose | Electricity/hardware if self-hosting at home |

I do not promise any free-tier hosting works: free tiers commonly lack the RAM, persistent disk, or background-process support this stack needs. Check a provider's current limits yourself.
You also need the **legal right to serve the book's content** (answers quote passages); a private/personal deployment is the safe default.

## Option A: local development
See [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Option B: Docker Compose self-hosting

```bash
cp .env.example .env            # edit SECRET_KEY etc.
mkdir -p data/books && cp /path/to/book.pdf data/books/selected_book.pdf
docker compose build
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2:3b
docker compose run --rm api python -m src.ingestion --detect-chapters-only
#   review data/config/chapters.json on the host, set "reviewed": true
docker compose run --rm api python -m src.ingestion
docker compose up -d api ui
```
Open <http://127.0.0.1:8501>. Re-index later: `docker compose stop api ui` → ingest → `docker compose up -d api ui`.
The embedding model downloads into the `hf_cache` volume on first ingestion (set `EMBEDDING_LOCAL_ONLY=true` afterwards if you wish).

Persistent volumes: `ollama_models` (LLM), `chroma_db` (vector index), `runtime` (**accounts DB**, active-index pointer, reports), `hf_cache`. `./data` is a bind mount (PDF and `chapters.json`).
`docker compose down` keeps volumes; `docker compose down -v` **deletes accounts and indexes**.
Build note: the Dockerfiles were not built in my environment; if a build fails, send the error.

### Backups
```bash
docker compose stop api
docker run --rm -v bookwise_runtime:/v -v "$PWD":/b alpine tar czf /b/runtime-backup.tgz -C /v .
docker run --rm -v bookwise_chroma_db:/v -v "$PWD":/b alpine tar czf /b/chroma-backup.tgz -C /v .
docker compose start api
```
(Volume names are prefixed by your project folder name; check `docker volume ls`.) The index can always be rebuilt from the PDF; **accounts cannot**, so `runtime` matters most. Back up `.env` (secret key) separately and securely: losing `SECRET_KEY` invalidates all sessions (users just log in again).

## Public exposure checklist

1. **HTTPS is mandatory**: tokens and passwords travel in requests. Put Caddy/nginx/Traefik in front, terminate TLS there, proxy to `127.0.0.1:8501` (UI) and keep the API private (the UI calls it over the internal network). Streamlit needs websocket proxying.
2. Keep Ollama's port **unpublished** (compose does not publish it).
3. Prefer `ALLOW_SIGNUP=false` or an `INVITE_CODE` for demos; accounts persist in `runtime/auth.sqlite3`.
4. Run **one** API worker/container: rate limiting is in-memory per process, and ChromaDB's persistent client is not designed for concurrent multi-process writers.
5. Firewall everything except 443; keep images/packages updated; rotate `SECRET_KEY` if leaked.
6. Remember inference is CPU/GPU heavy: three simultaneous questions on a small VM will be slow.

## Model / hardware trade-offs (rough, verify on your machine)

| Choice | Effect |
|---|---|
| `llama3.2:3b` (default) | Smallest sensible choice; roughly 2–3 GB model, ~4–6 GB RAM in use; CPU answers can take 10–60 s including verification. Weakest at judging claims. |
| Larger models (e.g. 7–8B class) | Better instruction-following and verification, ~8–16 GB RAM, slower on CPU. Change `OLLAMA_MODEL` and re-run `ollama pull`; re-test thresholds and the evaluation set. |
| `LLM_VERIFIER_MODE` | `single` (default): 2 LLM calls per question. `off`: 1 call, faster, less protection. `per_claim`: up to ~7 calls, slowest. |
| `FINAL_K` / `MAX_CONTEXT_CHARS` | Less context to read = faster prompt processing (defaults 4 / 4000). |
| Evidence-only mode | No generation at all: fastest and has no paraphrase risk. |
| GPU | Large speed-up; requires NVIDIA drivers/container toolkit (see commented block in compose). |
| `OLLAMA_NUM_CTX` | Larger context = more RAM; keep `MAX_CONTEXT_CHARS` (~6000 chars ≈ 1.5k tokens) consistent with it. |

Ollama keeps a model loaded for a few minutes after use, so the first question after idle time is slow.
