# BookWise 📖

Question-answering over **one fixed PDF book**, with chapter + PDF-page citations and a layered defence against unsupported answers.

**Stack:** Python 3.11 · FastAPI/Uvicorn · LangChain components · ChromaDB (local, persistent) · SentenceTransformers `all-MiniLM-L6-v2` ·
Ollama `llama3.2:3b` · Streamlit · pypdf · Pydantic · python-dotenv · Pytest. Auth extras: `argon2-cffi` (password hashing), `email-validator`, SQLite (accounts only), `requests`/`httpx`.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate        # Windows: see docs/SETUP_GUIDE.md
pip install -r requirements-dev.txt
cp .env.example .env                                          # set SECRET_KEY
ollama pull llama3.2:3b
cp /path/to/your-book.pdf data/books/selected_book.pdf
python -m src.ingestion --detect-chapters-only                # review data/config/chapters.json, set "reviewed": true
python -m src.ingestion
uvicorn api:app --port 8000                                   # terminal 1
streamlit run app.py                                          # terminal 2
python -m src.diagnose "your question" --ask                  # why was a passage kept/rejected? where does the time go?
pytest                                                        # offline tests
```

## Key behaviours
* Users cannot upload, choose, or modify the book; ingestion is an owner CLI only.
* Exact fallback: *"The selected book does not provide enough information to answer this question."* with no references.
* Outages (Ollama, ChromaDB, missing index) are HTTP 503/504 errors, never silent fallbacks.
* Not an unchanged Vercel app: see [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md).
* No guarantee of zero hallucinations: see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#the-grounding-pipeline-and-its-honest-limits).

## Docs
[Setup](docs/SETUP_GUIDE.md) · [Architecture](docs/ARCHITECTURE.md) · [Ingestion](docs/INGESTION_GUIDE.md) · [API](docs/API_GUIDE.md) · [Testing](docs/TESTING_GUIDE.md) · [Deployment](docs/DEPLOYMENT_GUIDE.md)

Your book is not included in this repository (copyrighted; git-ignored). The test book is an original fictional text.
