import hashlib
import logging
import sys
from collections.abc import MutableMapping
from typing import Any, TextIO

import structlog
from structlog.typing import Processor

_SENSITIVE_KEYS = frozenset(
    {"api_key", "x-api-key", "authorization", "password", "secret", "token", "openai_api_key"}
)


def redact_secrets(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Last line of defence: no value stored under a sensitive key reaches a log line."""
    for key in event_dict:
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = "[redacted]"
    return event_dict


def configure_logging(
    *, level: str = "INFO", json_output: bool = True, stream: TextIO | None = None
) -> None:
    stream = stream or sys.stdout
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        redact_secrets,
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=stream.isatty())
    )
    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(file=stream),
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(name)
    return logger


def content_fields(key: str, text: str, *, include_content: bool) -> dict[str, Any]:
    """Describe user text for a log line without recording it unless explicitly enabled."""
    fields: dict[str, Any] = {
        f"{key}_chars": len(text),
        f"{key}_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    }
    if include_content:
        fields[key] = text
    return fields
