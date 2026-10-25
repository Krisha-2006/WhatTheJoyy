"""Exception hierarchy.

Two families are kept strictly separate:

* Operational errors (model down, DB failure, index missing). These map to
  HTTP 5xx and are NEVER turned into "the book does not contain the answer".
* Insufficient evidence is NOT an exception; it is a normal pipeline outcome.
"""
from __future__ import annotations


class BookWiseError(Exception):
    code = "internal_error"
    http_status = 500
    user_message = "Something went wrong on the server. Please try again later."

    def __init__(self, detail: str | None = None, *, user_message: str | None = None):
        super().__init__(detail or self.user_message)
        self.detail = detail or self.user_message
        if user_message:
            self.user_message = user_message


# ---- operational (runtime) -------------------------------------------------
class ConfigError(BookWiseError):
    code = "configuration_error"
    user_message = "The server is not configured correctly. Please contact the owner."


class IndexNotReadyError(BookWiseError):
    code = "index_not_ready"
    http_status = 503
    user_message = "The book has not been indexed yet. The owner must run ingestion."


class VectorStoreError(BookWiseError):
    code = "vector_store_unavailable"
    http_status = 503
    user_message = "The book search index is temporarily unavailable. Please try again later."


class EmbeddingUnavailableError(BookWiseError):
    code = "embedding_model_unavailable"
    http_status = 503
    user_message = "The embedding model is unavailable. Please contact the owner."


class LLMUnavailableError(BookWiseError):
    code = "llm_unavailable"
    http_status = 503
    user_message = "The language model service (Ollama) is unavailable. Please try again later."


class LLMTimeoutError(BookWiseError):
    code = "llm_timeout"
    http_status = 504
    user_message = "The language model took too long to respond. Please try again."


# ---- authentication / abuse control ------------------------------------------
class AuthError(BookWiseError):
    code = "unauthorized"
    http_status = 401
    user_message = "Authentication required or credentials invalid."


class SignupDisabledError(BookWiseError):
    code = "signup_disabled"
    http_status = 403
    user_message = "Sign-up is currently closed."


class AccountError(BookWiseError):
    code = "account_error"
    http_status = 409
    user_message = "Unable to create an account with those details."


class RateLimitError(BookWiseError):
    code = "rate_limited"
    http_status = 429
    user_message = "Too many requests. Please wait a moment and try again."

    def __init__(self, retry_after: int = 30):
        super().__init__("rate limited")
        self.retry_after = retry_after


# ---- owner-side ingestion errors --------------------------------------------
class IngestionError(BookWiseError):
    code = "ingestion_error"


class PdfMissingError(IngestionError):
    code = "pdf_missing"


class PdfUnreadableError(IngestionError):
    code = "pdf_unreadable"


class NoExtractableTextError(IngestionError):
    code = "no_extractable_text"


class ChapterMappingError(IngestionError):
    code = "chapter_mapping_invalid"


class EmbeddingInputTooLongError(IngestionError):
    code = "embedding_input_too_long"


# ---- validation outcomes that are NOT operational ------------------------------
class InvalidModelOutputError(BookWiseError):
    """The LLM answered, but not in a valid structure. Handled as fallback."""

    code = "invalid_model_output"


class GroundingFailure(Exception):
    """Candidate answer failed a grounding check. Handled as fallback."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
