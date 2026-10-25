"""Email/password accounts in a SEPARATE SQLite file (never the vector DB).

* Passwords: Argon2id via argon2-cffi (salted, memory-hard).
* Sessions: opaque random tokens. Only an HMAC-SHA256 of each token (keyed with
  SECRET_KEY) is stored, so a leaked database cannot be replayed as tokens.
* Tokens expire; logout deletes the session row.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from src.errors import AccountError, AuthError, RateLimitError

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def validate_password_strength(password: str, min_length: int) -> None:
    if len(password) < min_length:
        raise AccountError(
            f"Password too short (minimum {min_length} characters).",
            user_message=f"Password must be at least {min_length} characters long.",
        )
    classes = sum(bool(re.search(p, password)) for p in (r"[a-z]", r"[A-Z]", r"\d", r"[^\w\s]"))
    if classes < 2 or password.lower() in {"password123", "1234567890", "qwertyuiop"}:
        raise AccountError(
            "Password too simple.",
            user_message="Please use a mix of letters, numbers or symbols.",
        )


class AuthStore:
    def __init__(self, db_path: Path, secret_key: str, session_ttl_hours: int = 12):
        self.db_path = Path(db_path)
        self._secret = secret_key.encode()
        self._ttl = timedelta(hours=session_ttl_hours)
        self._hasher = PasswordHasher()
        self._dummy_hash = self._hasher.hash("bookwise-dummy-password")  # equalises timing
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _token_hash(self, token: str) -> str:
        return hmac.new(self._secret, token.encode(), hashlib.sha256).hexdigest()

    # ------------------------------------------------------------ accounts
    def create_user(self, email: str, password: str, min_length: int = 10) -> int:
        validate_password_strength(password, min_length)
        email = email.strip().lower()
        try:
            with self._conn() as c:
                cur = c.execute(
                    "INSERT INTO users(email, password_hash, created_at) VALUES (?,?,?)",
                    (email, self._hasher.hash(password), _now().isoformat()),
                )
                return int(cur.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise AccountError("Email already registered.") from exc

    def authenticate(self, email: str, password: str) -> int:
        email = email.strip().lower()
        with self._conn() as c:
            row = c.execute("SELECT id, password_hash FROM users WHERE email=?", (email,)).fetchone()
        stored = row[1] if row else self._dummy_hash
        try:
            ok = self._hasher.verify(stored, password)
        except (VerificationError, InvalidHashError):
            ok = False
        if not ok or not row:
            raise AuthError("Invalid credentials.", user_message="Incorrect email or password.")
        if self._hasher.check_needs_rehash(stored):
            with self._conn() as c:
                c.execute("UPDATE users SET password_hash=? WHERE id=?", (self._hasher.hash(password), row[0]))
        return int(row[0])

    # ------------------------------------------------------------ sessions
    def create_session(self, user_id: int) -> tuple[str, datetime]:
        token = secrets.token_urlsafe(32)
        expires = _now() + self._ttl
        with self._conn() as c:
            c.execute("DELETE FROM sessions WHERE expires_at < ?", (_now().isoformat(),))
            c.execute(
                "INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES (?,?,?,?)",
                (self._token_hash(token), user_id, _now().isoformat(), expires.isoformat()),
            )
        return token, expires

    def user_for_token(self, token: Optional[str]) -> int:
        if not token:
            raise AuthError("Missing token.")
        with self._conn() as c:
            row = c.execute(
                "SELECT user_id, expires_at FROM sessions WHERE token_hash=?", (self._token_hash(token),)
            ).fetchone()
        if not row or datetime.fromisoformat(row[1]) < _now():
            raise AuthError("Invalid or expired token.")
        return int(row[0])

    def delete_session(self, token: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM sessions WHERE token_hash=?", (self._token_hash(token),))


class SlidingWindowLimiter:
    """In-memory per-key limiter. Per process: with several API workers each has
    its own counters (documented limitation)."""

    def __init__(self, limit: int, window_seconds: int = 60, clock=time.monotonic):
        self.limit, self.window, self._clock = limit, window_seconds, clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = self._clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                raise RateLimitError(retry_after=max(1, int(self.window - (now - q[0])) + 1))
            q.append(now)
