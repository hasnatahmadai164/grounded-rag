import json

import pytest
import structlog

from grounded_rag.logging import configure_logging, content_fields, get_logger


def test_json_output_contains_event_and_bound_context(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", json_output=True)
    structlog.contextvars.bind_contextvars(request_id="abc123")
    try:
        get_logger("test").info("query_answered", abstained=False)
    finally:
        structlog.contextvars.clear_contextvars()
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["event"] == "query_answered"
    assert line["request_id"] == "abc123"
    assert line["level"] == "info"


def test_sensitive_keys_are_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", json_output=True)
    get_logger("test").info("auth_attempt", api_key="grk_secret", authorization="Bearer x")
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["api_key"] == "[redacted]"
    assert line["authorization"] == "[redacted]"
    assert "grk_secret" not in json.dumps(line)


def test_content_is_described_but_not_recorded_by_default() -> None:
    fields = content_fields("question", "What is the notice period?", include_content=False)
    assert fields["question_chars"] == 26
    assert "question" not in fields


def test_content_is_recorded_when_enabled() -> None:
    fields = content_fields("question", "hello", include_content=True)
    assert fields["question"] == "hello"
