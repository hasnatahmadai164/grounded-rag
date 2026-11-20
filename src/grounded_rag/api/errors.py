from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from grounded_rag.errors import GroundedRagError
from grounded_rag.logging import get_logger

log = get_logger(__name__)

STATUS_BY_CODE: dict[str, int] = {
    "unauthenticated": 401,
    "missing_api_key": 401,
    "invalid_api_key": 401,
    "not_found": 404,
    "payload_too_large": 413,
    "invalid_request": 422,
    "unsupported_document": 422,
    "empty_document": 422,
    "request_timeout": 504,
    "llm_timeout": 504,
    "llm_error": 502,
    "llm_invalid_output": 502,
    "vector_store_error": 502,
    "dependency_unavailable": 503,
    "budget_exceeded": 503,
    "ingestion_failed": 500,
    "configuration_error": 500,
    "internal_error": 500,
}

# Server-side faults never echo their message: it may name internal hosts or settings.
_GENERIC_MESSAGE = {
    "configuration_error": "The service is not configured correctly.",
    "internal_error": "An unexpected error occurred.",
    "ingestion_failed": "An unexpected error occurred.",
    "llm_error": "The language model service failed to respond.",
    "vector_store_error": "The vector search service failed to respond.",
}

_HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}


def request_id_of(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


def error_response(
    request: Request,
    status: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "error": {"code": code, "message": message, "request_id": request_id_of(request)}
    }
    if details:
        body["error"]["details"] = details
    return JSONResponse(body, status_code=status)


async def handle_service_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, GroundedRagError)
    status = STATUS_BY_CODE.get(exc.code, 500)
    if status >= 500:
        log.error("request_failed", code=exc.code, detail=exc.message, exc_info=exc)
    else:
        log.info("request_rejected", code=exc.code, status=status)
    message = _GENERIC_MESSAGE.get(exc.code, exc.message)
    details = exc.details if status < 500 or exc.code == "dependency_unavailable" else None
    return error_response(request, status, exc.code, message, details)


async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Field paths and messages only: echoing the submitted values could repeat private text.
    problems = [
        {"field": ".".join(str(part) for part in error["loc"]), "problem": error["msg"]}
        for error in exc.errors()
    ]
    return error_response(
        request, 422, "invalid_request", "The request is not valid.", {"fields": problems}
    )


async def handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _HTTP_CODES.get(exc.status_code, "http_error")
    return error_response(request, exc.status_code, code, str(exc.detail))


async def handle_database_error(request: Request, exc: Exception) -> JSONResponse:
    log.error("database_unavailable", error=type(exc).__name__, exc_info=exc)
    return error_response(
        request, 503, "dependency_unavailable", "The database is temporarily unavailable."
    )


def unexpected_error_response(request: Request) -> JSONResponse:
    """The response for an exception nothing else handled. Built by the request middleware so
    it carries the request id like every other response."""
    return error_response(request, 500, "internal_error", _GENERIC_MESSAGE["internal_error"])


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(GroundedRagError, handle_service_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(SQLAlchemyError, handle_database_error)
    app.add_exception_handler(ConnectionError, handle_database_error)
