from typing import Any


class GroundedRagError(Exception):
    """Base class for every error the service raises on purpose.

    `code` is stable and part of the public error contract; messages are for humans.
    """

    code = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(GroundedRagError):
    code = "configuration_error"


class AuthenticationError(GroundedRagError):
    code = "unauthenticated"


class MissingApiKeyError(AuthenticationError):
    code = "missing_api_key"


class InvalidApiKeyError(AuthenticationError):
    code = "invalid_api_key"


class PayloadTooLargeError(GroundedRagError):
    code = "payload_too_large"


class NotFoundError(GroundedRagError):
    code = "not_found"


class InvalidRequestError(GroundedRagError):
    code = "invalid_request"


class UnsupportedDocumentError(GroundedRagError):
    code = "unsupported_document"


class EmptyDocumentError(UnsupportedDocumentError):
    code = "empty_document"


class IngestionError(GroundedRagError):
    code = "ingestion_failed"


class VectorStoreError(GroundedRagError):
    code = "vector_store_error"


class LLMError(GroundedRagError):
    code = "llm_error"


class LLMTimeoutError(LLMError):
    code = "llm_timeout"


class LLMOutputError(LLMError):
    code = "llm_invalid_output"


class BudgetExceededError(GroundedRagError):
    code = "budget_exceeded"


class RequestTimeoutError(GroundedRagError):
    code = "request_timeout"


class DependencyUnavailableError(GroundedRagError):
    code = "dependency_unavailable"
