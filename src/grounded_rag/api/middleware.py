import re
import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from grounded_rag.api.errors import error_response, unexpected_error_response
from grounded_rag.errors import PayloadTooLargeError
from grounded_rag.logging import get_logger

REQUEST_ID_HEADER = "x-request-id"
_ACCEPTABLE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

log = get_logger(__name__)


def resolve_request_id(inbound: str | None) -> str:
    """Keep a caller-supplied id when it is safe to log, otherwise mint one."""
    if inbound and _ACCEPTABLE_REQUEST_ID.match(inbound):
        return inbound
    return uuid.uuid4().hex


class RequestContextMiddleware:
    """Assigns a request id, echoes it in the response and writes one access log line."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        inbound = headers.get(REQUEST_ID_HEADER.encode(), b"").decode("latin-1") or None
        request_id = resolve_request_id(inbound)
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)

        status_code = 500
        response_started = False
        started = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                MutableHeaders(scope=message).append("X-Request-ID", request_id)
            await send(message)

        try:
            try:
                await self.app(scope, receive, send_with_request_id)
            except Exception as exc:
                log.error("request_crashed", exc_info=exc)
                if response_started:
                    raise
                response = unexpected_error_response(Request(scope))
                await response(scope, receive, send_with_request_id)
        finally:
            log.info(
                "request_completed",
                method=scope["method"],
                path=scope["path"],
                status=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            structlog.contextvars.clear_contextvars()


class BodySizeLimitMiddleware:
    """Rejects oversized request bodies, whether announced by Content-Length or streamed."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            response = error_response(
                Request(scope),
                413,
                PayloadTooLargeError.code,
                f"The request body may be at most {self.max_bytes} bytes.",
            )
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise PayloadTooLargeError(
                        f"The request body may be at most {self.max_bytes} bytes."
                    )
            return message

        await self.app(scope, limited_receive, send)
