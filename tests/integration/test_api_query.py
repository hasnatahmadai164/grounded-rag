import asyncio

import pytest

from grounded_rag.errors import LLMError
from tests.fakes import PurposeClient
from tests.integration.api_harness import api_harness

pytestmark = pytest.mark.db

GOVERNING = "This Agreement shall be governed by the laws of the State of Delaware."
SUFFICIENT = {"sufficient": True, "reason": "found", "missing": ""}
INSUFFICIENT = {"sufficient": False, "reason": "not there", "missing": "clause"}
ANSWER = {
    "answerable": True,
    "claims": [
        {
            "text": "Delaware law governs.",
            "citations": [{"passage_id": "P1", "quote": "the laws of the State of Delaware"}],
        }
    ],
}
QUESTION = {"question": "Which law governs the Supply Agreement?"}


async def seeded(api, roles=("legal",)):
    headers = await api.create_key("legal-analyst", set(roles))
    await api.ingest("Supply Agreement", f"13. Governing Law\n\n{GOVERNING}", {"legal"})
    return headers


async def test_a_supported_question_returns_a_cited_answer_with_usage(api) -> None:
    api.chat.replies.update(grade=[SUFFICIENT], generate=[ANSWER])
    headers = await seeded(api)

    response = await api.client.post("/v1/query", json=QUESTION, headers=headers)

    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == "Delaware law governs. [1]"
    assert body["abstained"] is False
    assert body["abstain_reason"] is None
    citation = body["citations"][0]
    assert citation["quote"] == "the laws of the State of Delaware"
    assert citation["document_title"] == "Supply Agreement"
    assert citation["section"] == "13. Governing Law"
    usage = body["usage"]
    assert usage["prompt_tokens"] == 400
    assert usage["completion_tokens"] == 80
    assert usage["cost_usd"] > 0
    assert usage["latency_ms"] > 0
    assert body["request_id"] == response.headers["x-request-id"]
    assert body["debug"] is None


async def test_the_quoted_offsets_locate_the_citation_in_the_stored_document(api) -> None:
    api.chat.replies.update(grade=[SUFFICIENT], generate=[ANSWER])
    headers = await seeded(api)

    body = (await api.client.post("/v1/query", json=QUESTION, headers=headers)).json()

    citation = body["citations"][0]
    doc_text = await api.container.documents.get_text(citation["doc_id"])
    assert doc_text is not None
    assert doc_text[citation["start_char"] : citation["end_char"]] == citation["quote"]


async def test_unsupported_questions_abstain_with_a_reason_and_are_not_errors(api) -> None:
    api.chat.replies.update(grade=[INSUFFICIENT] * 3, rewrite=[{"query": "a"}, {"query": "b"}])
    headers = await seeded(api)

    response = await api.client.post("/v1/query", json=QUESTION, headers=headers)

    body = response.json()
    assert response.status_code == 200
    assert body["abstained"] is True
    assert body["abstain_reason"] == "insufficient_evidence"
    assert body["citations"] == []


async def test_a_document_the_caller_cannot_see_behaves_like_one_that_does_not_exist(api) -> None:
    outsider = await api.create_key("finance-analyst", {"finance"})
    doc_id = await api.ingest("Supply Agreement", f"13. Governing Law\n\n{GOVERNING}", {"legal"})
    with_document = await api.client.post("/v1/query", json=QUESTION, headers=outsider)

    await api.container.ingestion.delete(doc_id)
    without_document = await api.client.post("/v1/query", json=QUESTION, headers=outsider)

    def comparable(response):
        body = response.json()
        return {k: v for k, v in body.items() if k not in {"request_id", "usage"}}

    assert with_document.status_code == without_document.status_code == 200
    assert comparable(with_document) == comparable(without_document)
    assert with_document.json()["abstain_reason"] == "no_relevant_context"
    assert api.chat.requests == []


async def test_debug_returns_the_trace_with_scores_for_every_stage(api) -> None:
    api.chat.replies.update(grade=[SUFFICIENT], generate=[ANSWER])
    headers = await seeded(api)

    body = (
        await api.client.post("/v1/query", json={**QUESTION, "debug": True}, headers=headers)
    ).json()

    debug = body["debug"]
    stages = debug["retrievals"][0]
    assert [s["stage"] for s in stages] == ["dense", "lexical", "fused", "final"]
    fused = next(s for s in stages if s["stage"] == "fused")
    assert fused["candidates"][0]["document_title"] == "Supply Agreement"
    assert fused["candidates"][0]["score"] > 0
    assert [e["node"] for e in debug["events"]][:2] == ["retrieve", "grade"]
    assert debug["model_calls"] == 2
    assert debug["queries"] == [QUESTION["question"]]


async def test_the_debug_trace_never_names_documents_the_caller_cannot_read(api) -> None:
    api.chat.replies.update(grade=[SUFFICIENT], generate=[ANSWER])
    headers = await seeded(api)
    await api.ingest("Revenue Memo", "Confidential revenue share of twelve percent.", {"finance"})

    body = (
        await api.client.post(
            "/v1/query",
            json={"question": "revenue share and governing law", "debug": True},
            headers=headers,
        )
    ).json()

    assert "Revenue Memo" not in str(body)
    assert "twelve percent" not in str(body)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "ok", "top_k": 0},
    ],
)
async def test_malformed_queries_are_rejected_before_any_work(api, payload) -> None:
    headers = await api.create_key("analyst", {"legal"})

    response = await api.client.post("/v1/query", json=payload, headers=headers)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert api.chat.requests == []


async def test_top_k_above_the_configured_maximum_is_rejected(api) -> None:
    headers = await api.create_key("analyst", {"legal"})

    response = await api.client.post("/v1/query", json={**QUESTION, "top_k": 21}, headers=headers)

    assert response.status_code == 422
    assert response.json()["error"]["details"] == {"max_top_k": 20}


async def test_querying_requires_a_key(api) -> None:
    response = await api.client.post("/v1/query", json=QUESTION)

    assert response.status_code == 401


async def test_a_model_failure_becomes_a_502_without_internal_detail(
    migrated_database_url, engine, tmp_path
) -> None:
    def explode(request):
        raise LLMError("OpenAI returned 500: upstream detail")

    chat = PurposeClient(grade=[explode])
    async with api_harness(migrated_database_url, tmp_path, chat) as api:
        headers = await seeded(api)

        response = await api.client.post("/v1/query", json=QUESTION, headers=headers)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_error"
    assert "upstream detail" not in response.text


async def test_a_request_that_takes_too_long_gets_a_504(
    migrated_database_url, engine, tmp_path
) -> None:
    class SlowChat(PurposeClient):
        async def complete(self, request):
            await asyncio.sleep(1)
            return await super().complete(request)

    async with api_harness(
        migrated_database_url, tmp_path, SlowChat(grade=[SUFFICIENT]), request_timeout_seconds=0.05
    ) as api:
        headers = await seeded(api)

        response = await api.client.post("/v1/query", json=QUESTION, headers=headers)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "request_timeout"
