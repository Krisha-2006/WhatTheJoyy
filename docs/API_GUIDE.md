# API Guide

Base URL (local): `http://127.0.0.1:8000` · Interactive docs: `/docs`.
Errors always look like `{"error": {"code": "...", "message": "..."}}` and never contain stack traces or internals.

| Method & path | Auth | Purpose |
|---|---|---|
| `POST /auth/signup` | – | Create account (`201`). May be disabled / invite-only. |
| `POST /auth/login` | – | Returns bearer token. |
| `POST /auth/logout` | Bearer | Invalidates the token (`204`). |
| `POST /ask` | Bearer | Ask one standalone question. |
| `GET /book` | – | Title, pages, chunk count, embedding model, index time. |
| `GET /health` | – | `200 ok` or `503 degraded` (index, Ollama reachable, model installed). |

There is deliberately **no** upload, ingestion, book-selection or delete endpoint.

## Auth

```bash
curl -X POST localhost:8000/auth/signup -H 'content-type: application/json' \
  -d '{"email":"me@example.com","password":"Correct-Horse-9","inviteCode":null}'
curl -X POST localhost:8000/auth/login -H 'content-type: application/json' \
  -d '{"email":"me@example.com","password":"Correct-Horse-9"}'
# → {"accessToken":"...","tokenType":"bearer","expiresAt":"2026-10-25T06:00:00+00:00"}
```
Passwords: ≥ `MIN_PASSWORD_LENGTH` (10) and at least two character classes.

## POST /ask

```json
{ "question": "Why does the character leave home?", "evidenceOnly": false }
```
Only `question` (≤ `MAX_QUESTION_CHARS`, default 500) and optional `evidenceOnly` are accepted. Any other field → `422`.

### Answered
```json
{
  "status": "answered",
  "answer": "Mirelle climbed 214 steps every evening to light the lamp. [1]",
  "references": [{
    "chunkId": "selected_book-p0003-00-1a2b3c4d",
    "chapter": "The Tide Tables",
    "section": null,
    "pdfPageNumber": 3,
    "printedPageLabel": "1",
    "excerpt": "Exact supporting book text…",
    "quote": "She climbed the 214 steps of the tower"
  }],
  "elapsedSeconds": 8.4
}
```
`quote` is the verbatim sentence of the excerpt that best matches the claim (chosen by code, not by the model). `elapsedSeconds` is server time for the whole question. `printedPageLabel` is `null` unless the PDF defines real page labels. `[n]` markers index into `references`.

### Insufficient evidence (HTTP 200)
```json
{ "status": "insufficient_evidence",
  "answer": "The selected book does not provide enough information to answer this question.",
  "references": [] }
```
Returned for: no/weak relevant evidence, evidence that doesn't answer, partial answers, invalid output, failed citation or grounding checks.

### Evidence-only mode
`evidenceOnly: true` skips the LLM: returns `status: "answered"`, a fixed explanatory sentence and the top relevant excerpts as references.
These are verbatim, policy-filtered passages; relevance is by similarity score, **not** judged by an LLM.

### Operational errors (never disguised as "insufficient evidence")

| HTTP | `code` | Meaning |
|---|---|---|
| 401 | `unauthorized` | Missing/invalid/expired token or bad credentials |
| 403 | `signup_disabled` | Signup closed or bad invite code |
| 409 | `account_error` | Weak password / email already used |
| 422 | `invalid_request`, `question_too_long` | Validation |
| 429 | `rate_limited` | `Retry-After` header set |
| 503 | `llm_unavailable` | Ollama unreachable or model not pulled |
| 503 | `vector_store_unavailable` | ChromaDB failure |
| 503 | `index_not_ready` | Owner has not run ingestion |
| 503 | `embedding_model_unavailable` | Embedding model cannot load |
| 504 | `llm_timeout` | Ollama too slow (`OLLAMA_TIMEOUT_SECONDS`) |
| 500 | `internal_error` | Unexpected; details only in server logs |

Chat history is **not** sent to `/ask`; every question is standalone and prior turns are never used as evidence.
