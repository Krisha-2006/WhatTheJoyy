"""Streamlit UI.  Run:  streamlit run app.py

The UI is a thin client: it only calls the FastAPI backend. There is no upload
widget, no book selector and no way to alter the book.
"""
from __future__ import annotations

import html
import os
from typing import Optional

import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.getenv("UI_REQUEST_TIMEOUT", "180"))

st.set_page_config(page_title="BookWise", page_icon="📖", layout="centered")


# ----------------------------------------------------------------- API helpers
class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def api(method: str, path: str, *, json: Optional[dict] = None, token: Optional[str] = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.request(method, f"{API}{path}", json=json, headers=headers, timeout=TIMEOUT)
    except requests.ConnectionError:
        raise ApiError(0, "backend_down", f"Cannot reach the BookWise backend at {API}. Is `uvicorn api:app` running?")
    except requests.Timeout:
        raise ApiError(0, "timeout", "The request timed out. The model may be busy; try again.")
    if r.status_code >= 400:
        try:
            err = r.json().get("error", {})
        except ValueError:
            err = {}
        raise ApiError(r.status_code, err.get("code", "error"), err.get("message", f"Server error ({r.status_code})."))
    return r.json() if r.content else None


@st.cache_data(ttl=300, show_spinner=False)
def book_info() -> Optional[dict]:
    try:
        return api("GET", "/book")
    except ApiError:
        return None


def page_display(ref: dict) -> str:
    text = f"PDF page {ref['pdfPageNumber']}"
    if ref.get("printedPageLabel"):
        text += f" (printed page {ref['printedPageLabel']})"
    return text


# ------------------------------------------------------------------ rendering
def render_excerpt(ref: dict) -> None:
    excerpt = ref["excerpt"]
    safe = html.escape(excerpt)
    quote = ref.get("quote")
    if quote and quote in excerpt:
        safe = html.escape(excerpt).replace(html.escape(quote), f"<mark>{html.escape(quote)}</mark>", 1)
    st.html(f"<blockquote style='margin:0;padding-left:.8em;border-left:3px solid #999'>{safe}</blockquote>")


def render_response(resp: dict) -> None:
    if resp["status"] == "insufficient_evidence":
        st.info(resp["answer"])
        return
    st.write(resp["answer"])
    if resp.get("elapsedSeconds") is not None:
        st.caption(f"Answered in {resp['elapsedSeconds']:.1f}s")
    refs = resp.get("references", [])
    if refs:
        st.caption("Sources (exact text from the book)")
    for i, ref in enumerate(refs, start=1):
        section = f" › {ref['section']}" if ref.get("section") else ""
        label = f"[{i}] {ref['chapter']}{section} — {page_display(ref)}"
        with st.expander(label):
            render_excerpt(ref)


# ------------------------------------------------------------------ auth views
def auth_view() -> None:
    info = book_info()
    st.title("📖 BookWise")
    if info:
        st.subheader(info["title"])
    st.caption("Ask questions about this one book. Answers come only from the book and cite chapter and page.")
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])
    with tab_login:
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                try:
                    data = api("POST", "/auth/login", json={"email": email, "password": password})
                    st.session_state.update(token=data["accessToken"], email=email, history=[])
                    st.rerun()
                except ApiError as e:
                    st.error(e.message)
    with tab_signup:
        with st.form("signup"):
            email = st.text_input("Email", key="su_email")
            password = st.text_input("Password (10+ characters)", type="password", key="su_pw")
            invite = st.text_input("Invite code (if required)", key="su_invite")
            if st.form_submit_button("Create account"):
                try:
                    api("POST", "/auth/signup", json={"email": email, "password": password, "inviteCode": invite or None})
                    st.success("Account created. Switch to the Log in tab.")
                except ApiError as e:
                    st.error(e.message)


# ------------------------------------------------------------------- chat view
def chat_view() -> None:
    info = book_info()
    with st.sidebar:
        st.write(f"Signed in as **{st.session_state.get('email', '')}**")
        evidence_only = st.checkbox(
            "Evidence-only mode",
            help="Fast: shows the most relevant book excerpts without an AI-written explanation (no language-model call).",
        )
        if st.button("Log out"):
            try:
                api("POST", "/auth/logout", token=st.session_state.get("token"))
            except ApiError:
                pass
            st.session_state.clear()
            st.rerun()

    st.title("📖 BookWise")
    if info:
        st.subheader(info["title"])
    st.caption("Each question is answered on its own; earlier messages are not used as evidence.")

    for turn in st.session_state.get("history", []):
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            if "error" in turn:
                st.error(turn["error"])
            else:
                render_response(turn["resp"])

    question = st.chat_input("Ask a question about the book…", max_chars=500)
    if not question:
        return
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the book and checking the answer… (the first question after a pause can take longer while the model loads)"):
                resp = api(
                    "POST", "/ask",
                    json={"question": question, "evidenceOnly": evidence_only},
                    token=st.session_state["token"],
                )
            render_response(resp)
            st.session_state["history"].append({"q": question, "resp": resp})
        except ApiError as e:
            if e.status == 401:
                st.session_state.clear()
                st.session_state["notice"] = "Your session expired. Please log in again."
                st.rerun()
            elif e.status == 429:
                msg = "You are asking too quickly. Please wait a few seconds and try again."
            elif e.code in {"llm_unavailable", "llm_timeout"}:
                msg = f"{e.message} (Owner tip: make sure `ollama serve` is running and the model is pulled.)"
            elif e.code == "index_not_ready":
                msg = f"{e.message} (Owner: run `python -m src.ingestion`.)"
            else:
                msg = e.message
            st.error(msg)
            st.session_state["history"].append({"q": question, "error": msg})


def main() -> None:
    if notice := st.session_state.pop("notice", None):
        st.warning(notice)
    if st.session_state.get("token"):
        chat_view()
    else:
        auth_view()


main()
