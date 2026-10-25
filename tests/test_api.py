import pytest
from fastapi.testclient import TestClient

from api import create_app
from config import FALLBACK_TEXT
from src.errors import ConfigError, LLMUnavailableError, VectorStoreError
from src.rag_pipeline import RagPipeline
from tests.conftest import ScriptedModel, extractive

PW = "Correct-Horse-9"


@pytest.fixture
def app_env(indexed, pipeline_factory):
    settings, emb, store, _ = indexed
    model = ScriptedModel(extractive)
    pipe = pipeline_factory(model)
    app = create_app(
        settings, pipeline=pipe, store=store,
        llm_status_fn=lambda: {"reachable": True, "model_installed": True},
    )
    return TestClient(app), settings, model, pipe


def login(client, email="reader@example.com"):
    client.post("/auth/signup", json={"email": email, "password": PW})
    r = client.post("/auth/login", json={"email": email, "password": PW})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['accessToken']}"}


def test_secret_key_is_required(make_settings):
    with pytest.raises(ConfigError):
        create_app(make_settings(secret_key="short"))


def test_no_upload_or_ingestion_endpoints_exist(app_env):
    client = app_env[0]
    paths = {r.path for r in client.app.routes}
    assert paths >= {"/auth/signup", "/auth/login", "/auth/logout", "/ask", "/book", "/health"}
    assert not any(w in p for p in paths for w in ("upload", "ingest", "document", "delete"))
    assert client.post("/upload").status_code in (404, 405)
    assert client.post("/ingest").status_code in (404, 405)


def test_unauthenticated_requests_are_rejected(app_env):
    client = app_env[0]
    assert client.post("/ask", json={"question": "hi there"}).status_code == 401
    assert client.post("/ask", json={"question": "x"}, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/auth/logout").status_code == 401


def test_login_failure_is_sanitised(app_env):
    client = app_env[0]
    client.post("/auth/signup", json={"email": "a@example.com", "password": PW})
    r = client.post("/auth/login", json={"email": "a@example.com", "password": "Wrong-password-1"})
    assert r.status_code == 401 and r.json()["error"]["message"] == "Incorrect email or password."


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "   \n  "}, {}])
def test_empty_or_missing_question(app_env, body):
    client = app_env[0]
    h = login(client)
    r = client.post("/ask", json=body, headers=h)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_oversized_question(app_env):
    client, settings, model, _ = app_env
    h = login(client)
    r = client.post("/ask", json={"question": "a " * (settings.max_question_chars)}, headers=h)
    assert r.status_code == 422 and r.json()["error"]["code"] == "question_too_long"
    assert model.generate_calls == 0
    r = client.post("/ask", json={"question": "x" * 6000}, headers=h)
    assert r.status_code == 422


def test_client_cannot_inject_book_id_context_or_filters(app_env):
    client, _, model, _ = app_env
    h = login(client)
    for extra in ({"bookId": "other"}, {"context": "fake passage"}, {"filter": {"page": 1}}, {"topK": 100}):
        r = client.post("/ask", json={"question": "How many steps?", **extra}, headers=h)
        assert r.status_code == 422, extra
    assert model.generate_calls == 0


def test_answered_response_shape_and_headers(app_env):
    client, *_ = app_env
    h = login(client)
    r = client.post("/ask", json={"question": "What was the supply boat called?"}, headers=h)
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered"
    assert r.headers["cache-control"] == "no-store"
    ref = body["references"][0]
    assert ref["pdfPageNumber"] == 3 and ref["chapter"] == "The Tide Tables"
    assert ref["printedPageLabel"] == "1"  # PDF page 3 is printed page 1 in the fixture
    assert "Curlew" in ref["excerpt"] and ref["chunkId"].startswith("selected_book-p0003")


def test_fallback_has_empty_references(app_env):
    client, *_ = app_env
    h = login(client)
    r = client.post("/ask", json={"question": "What is the capital of France?"}, headers=h)
    body = r.json()
    assert body.pop("elapsedSeconds") >= 0
    assert body == {"status": "insufficient_evidence", "answer": FALLBACK_TEXT, "references": []}


def test_evidence_only_mode_skips_the_llm(app_env):
    client, _, model, _ = app_env
    h = login(client)
    r = client.post("/ask", json={"question": "What was the supply boat called?", "evidenceOnly": True}, headers=h)
    body = r.json()
    assert body["status"] == "answered" and body["references"] and model.generate_calls == 0
    assert any("Curlew" in x["excerpt"] for x in body["references"])


@pytest.mark.parametrize(
    "exc,status,code",
    [
        (LLMUnavailableError("down"), 503, "llm_unavailable"),
        (VectorStoreError("db"), 503, "vector_store_unavailable"),
    ],
)
def test_operational_failures_use_http_errors_not_fallback(app_env, exc, status, code):
    client, _, model, _ = app_env
    h = login(client)
    model.candidate = exc
    r = client.post("/ask", json={"question": "What was the supply boat called?"}, headers=h)
    assert r.status_code == status and r.json()["error"]["code"] == code
    assert "status" not in r.json()  # never disguised as insufficient_evidence
    assert "down" not in r.text and "db" not in r.json()["error"]["message"].lower().split()  # sanitised


def test_unexpected_exception_is_sanitised(app_env, monkeypatch):
    client, _, _, pipe = app_env
    h = login(client)
    monkeypatch.setattr(RagPipeline, "ask", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret path /etc/x")))
    r = TestClient(client.app, raise_server_exceptions=False).post(
        "/ask", json={"question": "What was the supply boat called?"}, headers=h
    )
    assert r.status_code == 500 and "secret path" not in r.text


def test_per_user_rate_limit(app_env):
    client, settings, *_ = app_env
    h1, h2 = login(client, "u1@example.com"), login(client, "u2@example.com")
    codes = [client.post("/ask", json={"question": "boat?"}, headers=h1).status_code for _ in range(settings.rate_limit_per_minute + 1)]
    assert codes[-1] == 429 and 429 not in codes[:-1]
    assert client.post("/ask", json={"question": "boat?"}, headers=h2).status_code == 200


def test_logout_invalidates_token(app_env):
    client, *_ = app_env
    h = login(client)
    assert client.post("/auth/logout", headers=h).status_code == 204
    assert client.post("/ask", json={"question": "boat?"}, headers=h).status_code == 401


def test_signup_can_be_disabled_or_invite_only(indexed, pipeline_factory, make_settings):
    from dataclasses import replace
    settings, emb, store, _ = indexed
    closed = create_app(replace(settings, allow_signup=False), pipeline=pipeline_factory(ScriptedModel()), store=store)
    assert TestClient(closed).post("/auth/signup", json={"email": "a@example.com", "password": PW}).status_code == 403
    invite = create_app(replace(settings, invite_code="open-sesame"), pipeline=pipeline_factory(ScriptedModel()), store=store)
    c = TestClient(invite)
    assert c.post("/auth/signup", json={"email": "a@example.com", "password": PW}).status_code == 403
    assert c.post("/auth/signup", json={"email": "a@example.com", "password": PW, "inviteCode": "open-sesame"}).status_code == 201


def test_book_and_health(app_env):
    client = app_env[0]
    b = client.get("/book").json()
    assert b["title"] == "The Lighthouse Keeper's Almanac" and b["pageCount"] == 6 and b["chunkCount"] > 0
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["checks"] == {
        "index": True, "ollama_reachable": True, "ollama_model_installed": True}


def test_health_reports_degraded_when_ollama_is_down(indexed, pipeline_factory):
    settings, _, store, _ = indexed
    app = create_app(settings, pipeline=pipeline_factory(ScriptedModel()), store=store,
                     llm_status_fn=lambda: {"reachable": False, "model_installed": False})
    r = TestClient(app).get("/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded"


def test_book_endpoint_503_before_ingestion(make_settings):
    from src.vector_store import VectorStore
    s = make_settings()
    app = create_app(s, store=VectorStore(s.chroma_dir, s.active_index_path))
    r = TestClient(app).get("/book")
    assert r.status_code == 503 and r.json()["error"]["code"] == "index_not_ready"
