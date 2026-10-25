import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.auth import AuthStore, SlidingWindowLimiter
from src.errors import AccountError, AuthError, RateLimitError

SECRET = "s" * 48
GOOD_PW = "Correct-Horse-9"


@pytest.fixture
def store(tmp_path):
    return AuthStore(tmp_path / "auth.sqlite3", SECRET, session_ttl_hours=1)


def test_signup_login_logout_cycle(store):
    uid = store.create_user("Reader@Example.com", GOOD_PW)
    assert store.authenticate("reader@example.com", GOOD_PW) == uid  # email is case-insensitive
    token, _ = store.create_session(uid)
    assert store.user_for_token(token) == uid
    store.delete_session(token)
    with pytest.raises(AuthError):
        store.user_for_token(token)


def test_wrong_password_and_unknown_user_give_same_error(store):
    store.create_user("a@example.com", GOOD_PW)
    with pytest.raises(AuthError) as e1:
        store.authenticate("a@example.com", "wrong-password-1")
    with pytest.raises(AuthError) as e2:
        store.authenticate("nobody@example.com", GOOD_PW)
    assert e1.value.user_message == e2.value.user_message == "Incorrect email or password."


def test_no_plaintext_password_or_raw_token_is_stored(store, tmp_path):
    uid = store.create_user("a@example.com", GOOD_PW)
    token, _ = store.create_session(uid)
    raw = sqlite3.connect(tmp_path / "auth.sqlite3")
    pw_hash = raw.execute("SELECT password_hash FROM users").fetchone()[0]
    stored_token = raw.execute("SELECT token_hash FROM sessions").fetchone()[0]
    dump = "\n".join(raw.iterdump())
    assert pw_hash.startswith("$argon2") and GOOD_PW not in dump
    assert token not in dump and stored_token != token


def test_duplicate_email_and_weak_passwords_rejected(store):
    store.create_user("a@example.com", GOOD_PW)
    with pytest.raises(AccountError):
        store.create_user("A@example.com", GOOD_PW)
    with pytest.raises(AccountError):
        store.create_user("b@example.com", "short1A")
    with pytest.raises(AccountError):
        store.create_user("c@example.com", "alllowercaseletters")


def test_expired_session_is_rejected(store, tmp_path):
    uid = store.create_user("a@example.com", GOOD_PW)
    token, _ = store.create_session(uid)
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    with sqlite3.connect(tmp_path / "auth.sqlite3") as c:
        c.execute("UPDATE sessions SET expires_at=?", (past,))
    with pytest.raises(AuthError):
        store.user_for_token(token)


def test_token_from_other_secret_is_not_valid(store, tmp_path):
    uid = store.create_user("a@example.com", GOOD_PW)
    token, _ = store.create_session(uid)
    other = AuthStore(tmp_path / "auth.sqlite3", "z" * 48)
    with pytest.raises(AuthError):
        other.user_for_token(token)


def test_sliding_window_rate_limiter():
    now = [0.0]
    lim = SlidingWindowLimiter(2, 60, clock=lambda: now[0])
    lim.check("u1"); lim.check("u1")
    with pytest.raises(RateLimitError) as e:
        lim.check("u1")
    assert e.value.retry_after >= 1
    lim.check("u2")  # other users are unaffected
    now[0] = 61
    lim.check("u1")  # window expired
