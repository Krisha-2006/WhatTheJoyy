"""FastAPI backend.  Run:  uvicorn api:app --host 127.0.0.1 --port 8000"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import Settings, get_settings
from src.auth import AuthStore, SlidingWindowLimiter
from src.embeddings import get_embedder
from src.errors import (
    AuthError,
    BookWiseError,
    ConfigError,
    IndexNotReadyError,
    SignupDisabledError,
)
from src.generation import OllamaAnswerModel
from src.rag_pipeline import RagPipeline
from src.retrieval import RetrievalPolicy, Retriever
from src.schemas import (
    AskRequest,
    AskResponse,
    BookInfo,
    LoginRequest,
    MessageResponse,
    SignupRequest,
    TokenResponse,
)
from src.vector_store import VectorStore

log = logging.getLogger("bookwise.api")
bearer = HTTPBearer(auto_error=False)


def build_default_pipeline(settings: Settings, store: VectorStore) -> RagPipeline:
    embedder = get_embedder(settings.embedding_model, settings.embedding_local_only)
    retriever = Retriever(RetrievalPolicy.from_settings(settings), embedder, store)
    return RagPipeline(settings, retriever, OllamaAnswerModel(settings))


def create_app(
    settings: Optional[Settings] = None,
    *,
    pipeline: Optional[RagPipeline] = None,
    auth_store: Optional[AuthStore] = None,
    store: Optional[VectorStore] = None,
    llm_status_fn=None,
) -> FastAPI:
    settings = settings or get_settings()
    settings.validate_for_api()
    store = store or VectorStore(settings.chroma_dir, settings.active_index_path)
    auth = auth_store or AuthStore(settings.auth_db_path, settings.secret_key, settings.session_ttl_hours)
    ask_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute, 60)
    login_limiter = SlidingWindowLimiter(settings.login_attempts_per_minute, 60)
    signup_limiter = SlidingWindowLimiter(5, 3600)
    holder: dict = {"pipeline": pipeline}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Make BookWise's own INFO logs (retrieval decisions, timings) visible.
        bw = logging.getLogger("bookwise")
        if not bw.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            bw.addHandler(h)
        bw.setLevel(logging.INFO)
        if settings.warmup_on_start and pipeline is None:
            # background thread: the API accepts requests immediately
            threading.Thread(target=lambda: get_pipeline().warmup(), daemon=True).start()
        yield

    app = FastAPI(title="BookWise API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings

    def get_pipeline() -> RagPipeline:
        if holder["pipeline"] is None:
            holder["pipeline"] = build_default_pipeline(settings, store)
        return holder["pipeline"]

    # ------------------------------------------------------------ errors
    @app.exception_handler(BookWiseError)
    async def _bookwise_error(_: Request, exc: BookWiseError):
        if exc.http_status >= 500:
            log.error("operational error %s: %s", exc.code, exc.detail)
        headers = {"Retry-After": str(getattr(exc, "retry_after", ""))} if exc.http_status == 429 else None
        if exc.http_status == 401:
            headers = {"WWW-Authenticate": "Bearer"}
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": {"code": exc.code, "message": exc.user_message}},
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        msgs = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", []) if x != "body")
            msgs.append(f"{loc}: {err.get('msg', 'invalid')}".strip(": "))
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": "; ".join(msgs)[:300]}},
        )

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Something went wrong on the server."}},
        )

    # ------------------------------------------------------------ auth dep
    def current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> tuple[int, str]:
        if creds is None or creds.scheme.lower() != "bearer":
            raise AuthError("Missing bearer token.")
        return auth.user_for_token(creds.credentials), creds.credentials

    # ------------------------------------------------------------ routes
    @app.post("/auth/signup", status_code=201, response_model=MessageResponse)
    def signup(body: SignupRequest, request: Request):
        if not settings.allow_signup:
            raise SignupDisabledError("ALLOW_SIGNUP=false")
        signup_limiter.check(request.client.host if request.client else "unknown")
        if settings.invite_code and body.invite_code != settings.invite_code:
            raise SignupDisabledError("bad invite code", user_message="A valid invite code is required.")
        auth.create_user(body.email, body.password, settings.min_password_length)
        return MessageResponse(message="Account created. You can now log in.")

    @app.post("/auth/login", response_model=TokenResponse)
    def login(body: LoginRequest, request: Request):
        ip = request.client.host if request.client else "unknown"
        login_limiter.check(f"{ip}:{body.email.lower()}")
        user_id = auth.authenticate(body.email, body.password)
        token, expires = auth.create_session(user_id)
        return TokenResponse(access_token=token, expires_at=expires.isoformat())

    @app.post("/auth/logout", status_code=204)
    def logout(user=Depends(current_user)):
        auth.delete_session(user[1])
        return Response(status_code=204)

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, user=Depends(current_user)):
        user_id = user[0]
        ask_limiter.check(f"user:{user_id}")
        if len(body.question) > settings.max_question_chars:
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "code": "question_too_long",
                        "message": f"Questions are limited to {settings.max_question_chars} characters.",
                    }
                },
            )
        response = get_pipeline().ask(body.question, evidence_only=body.evidence_only)
        # Per-user content must never be shared/cached by proxies or browsers.
        return JSONResponse(content=response.model_dump(by_alias=True), headers={"Cache-Control": "no-store"})

    @app.get("/book", response_model=BookInfo)
    def book():
        active = store.active()  # raises IndexNotReadyError -> 503
        return BookInfo(
            title=settings.book_title_override or active.book_title,
            author=active.author,
            page_count=active.page_count,
            chunk_count=active.chunk_count,
            embedding_model=active.embedding_model,
            indexed_at=active.indexed_at,
        )

    @app.get("/health")
    def health(response: Response):
        index_ok = store.ping()
        llm = (llm_status_fn or (lambda: OllamaAnswerModel(settings).status()))()
        ok = index_ok and llm.get("reachable", False) and llm.get("model_installed", False)
        if not ok:
            response.status_code = 503
        return {
            "status": "ok" if ok else "degraded",
            "checks": {
                "index": index_ok,
                "ollama_reachable": llm.get("reachable", False),
                "ollama_model_installed": llm.get("model_installed", False),
            },
        }

    return app


def __getattr__(name: str):  # lets `uvicorn api:app` work without import-time side effects
    if name == "app":
        global app  # noqa: PLW0603
        app = create_app()
        return app
    raise AttributeError(name)
