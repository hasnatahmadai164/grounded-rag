import json

import pytest

from grounded_rag.logging import configure_logging

pytestmark = pytest.mark.db

SUFFICIENT = {"sufficient": True, "reason": "found", "missing": ""}
ANSWER = {
    "answerable": True,
    "claims": [
        {
            "text": "Delaware law governs.",
            "citations": [{"passage_id": "P1", "quote": "the laws of the State of Delaware"}],
        }
    ],
}
QUESTION = "Which law governs the Supply Agreement? (confidential wording 7731)"


def log_lines(captured: str) -> list[dict]:
    return [json.loads(line) for line in captured.splitlines() if line.startswith("{")]


async def ask(api):
    api.chat.replies.update(grade=[SUFFICIENT], generate=[ANSWER])
    headers = await api.create_key("legal-analyst", {"legal"})
    await api.ingest(
        "Supply Agreement",
        "This Agreement is governed by the laws of the State of Delaware.",
        {"legal"},
    )
    response = await api.client.post("/v1/query", json={"question": QUESTION}, headers=headers)
    return headers, response


async def test_every_log_line_of_a_request_carries_its_request_id_and_principal(
    api, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(level="INFO", json_output=True)
    _, response = await ask(api)

    lines = [
        line
        for line in log_lines(capsys.readouterr().out)
        if line.get("request_id") == response.headers["x-request-id"]
    ]

    events = {line["event"] for line in lines}
    assert {"query_received", "question_answered", "request_completed"} <= events
    completed = next(line for line in lines if line["event"] == "request_completed")
    assert completed["status"] == 200
    assert completed["principal"] == "legal-analyst"


async def test_logs_never_contain_the_api_key_or_the_question_by_default(
    api, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(level="DEBUG", json_output=True)
    headers, _ = await ask(api)

    output = capsys.readouterr().out

    assert headers["X-API-Key"] not in output
    assert "confidential wording 7731" not in output
    assert "Delaware law governs" not in output


async def test_content_is_logged_only_when_explicitly_enabled(
    migrated_database_url, engine, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.integration.api_harness import api_harness

    configure_logging(level="INFO", json_output=True)
    async with api_harness(migrated_database_url, tmp_path, log_content=True) as api:
        await ask(api)

    received = [e for e in log_lines(capsys.readouterr().out) if e["event"] == "query_received"]
    assert received[0]["question"] == QUESTION
