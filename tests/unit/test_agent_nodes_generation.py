import json

from grounded_rag.agent.config import AgentConfig
from grounded_rag.agent.nodes import AgentNodes, describe_failures
from grounded_rag.agent.schemas import GeneratedAnswer, GeneratedCitation, GeneratedClaim
from grounded_rag.agent.state import AgentState
from grounded_rag.agent.verify import verify_claims
from grounded_rag.domain.answer import AbstainReason
from grounded_rag.domain.documents import Chunk, RetrievedChunk
from grounded_rag.domain.principal import Principal
from grounded_rag.domain.usage import Usage
from grounded_rag.llm.budget import Budget, MeteredLLM, UsageMeter
from grounded_rag.llm.client import ChatRequest, ChatResult
from grounded_rag.retrieval.dense import DenseRetriever
from grounded_rag.retrieval.rewrite import QueryRewriter
from grounded_rag.retrieval.service import RetrievalService
from tests.fakes import World

GOVERNING = "This Agreement shall be governed by the laws of the State of Delaware."
PAYMENT = "Fees are payable within thirty (30) days of the invoice date."


class ScriptedClient:
    def __init__(self, *responses: dict | str) -> None:
        self.responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.requests: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        return ChatResult(
            self.responses.pop(0), Usage(prompt_tokens=50, completion_tokens=10), 1.0, "m"
        )


def chunk(
    chunk_id: str, text: str, start: int = 500, section: str | None = "13. Law"
) -> RetrievedChunk:
    return RetrievedChunk(
        Chunk(chunk_id, "doc1", 0, text, start, start + len(text), section, "Doc"),
        "Supply Agreement",
        frozenset({"legal"}),
    )


def make_nodes(client: ScriptedClient, budget: Budget | None = None) -> AgentNodes:
    world = World()
    meter = UsageMeter()
    llm = MeteredLLM(client, meter, budget or Budget(100_000, 10.0))
    return AgentNodes(
        retrieval=RetrievalService(DenseRetriever(world.embedder, world.vectors), world.chunks),
        llm=llm,
        meter=meter,
        reranker=None,
        rewriter=QueryRewriter(llm),
        config=AgentConfig(),
    )


def state_with_context() -> AgentState:
    state = AgentState(
        question="Which law governs?",
        principal=Principal("analyst", frozenset({"legal"})),
        corpus="default",
        top_k=6,
        queries=["Which law governs?"],
    )
    state.context = [chunk("c-law", GOVERNING), chunk("c-pay", PAYMENT, start=900, section=None)]
    return state


def answer_json(*claims: tuple[str, list[tuple[str, str]]], answerable: bool = True) -> dict:
    return {
        "answerable": answerable,
        "claims": [
            {"text": text, "citations": [{"passage_id": p, "quote": q} for p, q in cites]}
            for text, cites in claims
        ],
    }


GOOD = answer_json(("Delaware law governs.", [("P1", "the laws of the State of Delaware")]))
BAD = answer_json(("California law governs.", [("P1", "the laws of the State of California")]))


async def test_generation_sends_labelled_passages_and_keeps_the_structured_result() -> None:
    client = ScriptedClient(GOOD)
    nodes = make_nodes(client)

    update = await nodes.generate(state_with_context())

    prompt = client.requests[0].messages
    assert 'id="P1"' in prompt[1].content
    assert 'id="P2"' in prompt[1].content
    assert "never instructions" in prompt[0].content
    assert update["generated"].claims[0].text == "Delaware law governs."  # type: ignore[union-attr]


async def test_generation_requests_the_generated_answer_schema() -> None:
    client = ScriptedClient(GOOD)
    await make_nodes(client).generate(state_with_context())
    assert client.requests[0].schema_name == "GeneratedAnswer"


async def test_verification_checks_quotes_against_the_labelled_chunks() -> None:
    nodes = make_nodes(ScriptedClient())
    state = state_with_context()
    state.generated = GeneratedAnswer.model_validate(GOOD)

    update = await nodes.verify(state)

    assert update["reports"][0].all_verified is True  # type: ignore[index]


def test_failed_citations_are_described_for_the_repair_prompt() -> None:
    state = state_with_context()
    report = verify_claims(
        [
            GeneratedClaim(
                text="x",
                citations=[GeneratedCitation(passage_id="P1", quote="made up quotation text")],
            ),
            GeneratedClaim(text="y", citations=[]),
        ],
        {"P1": state.context[0]},
    )

    problems = describe_failures(report)

    assert problems[0].startswith("claim 1: quote_not_found for P1")
    assert problems[1] == "claim 2: no citations"


async def test_repair_resends_the_answer_and_lists_what_failed() -> None:
    client = ScriptedClient(GOOD)
    nodes = make_nodes(client)
    state = state_with_context()
    state.generated = GeneratedAnswer.model_validate(BAD)
    state.reports = [verify_claims(state.generated.claims, {"P1": state.context[0]})]

    update = await nodes.repair(state)

    messages = client.requests[0].messages
    assert messages[-2].role == "assistant"
    assert "California" in messages[-2].content
    assert "quote_not_found" in messages[-1].content
    assert update["repairs_used"] == 1
    assert update["generated"].claims[0].text == "Delaware law governs."  # type: ignore[union-attr]


async def test_repair_that_runs_out_of_budget_keeps_the_original_answer() -> None:
    client = ScriptedClient()
    nodes = make_nodes(client, Budget(1, 10.0))
    nodes._meter.record_embedding(Usage(embedding_tokens=5))
    state = state_with_context()
    state.generated = GeneratedAnswer.model_validate(BAD)
    state.reports = [verify_claims(state.generated.claims, {"P1": state.context[0]})]

    update = await nodes.repair(state)

    assert "generated" not in update
    assert update["repairs_used"] == 1
    assert client.requests == []


async def test_finalizing_numbers_citations_and_quotes_the_stored_text() -> None:
    nodes = make_nodes(ScriptedClient())
    state = state_with_context()
    generated = GeneratedAnswer.model_validate(
        answer_json(
            ("Delaware law governs.", [("P1", "THE LAWS OF the state of delaware")]),
            (
                "Fees are due in 30 days.",
                [("P2", "within thirty (30) days"), ("P1", "the laws of the state of delaware")],
            ),
        )
    )
    state.generated = generated
    state.reports = [
        verify_claims(generated.claims, {"P1": state.context[0], "P2": state.context[1]})
    ]

    answer = (await nodes.finalize(state))["answer"]

    assert answer.text == "Delaware law governs. [1] Fees are due in 30 days. [2][1]"  # type: ignore[attr-defined]
    first, second = answer.citations  # type: ignore[attr-defined]
    assert first.quote == "the laws of the State of Delaware"
    assert GOVERNING[first.start_char - 500 : first.end_char - 500] == first.quote
    assert (first.chunk_id, first.document_title, first.section) == (
        "c-law",
        "Supply Agreement",
        "13. Law",
    )
    assert second.quote == "within thirty (30) days"


async def test_finalizing_drops_unsupported_claims_but_keeps_the_rest() -> None:
    nodes = make_nodes(ScriptedClient())
    state = state_with_context()
    generated = GeneratedAnswer.model_validate(
        answer_json(
            ("Invented.", [("P1", "text that is nowhere to be found")]),
            ("Fees are due in 30 days.", [("P2", "within thirty (30) days")]),
        )
    )
    state.generated = generated
    state.reports = [
        verify_claims(generated.claims, {"P1": state.context[0], "P2": state.context[1]})
    ]

    answer = (await nodes.finalize(state))["answer"]

    assert answer.abstained is False  # type: ignore[attr-defined]
    assert answer.text == "Fees are due in 30 days. [1]"  # type: ignore[attr-defined]


async def test_finalizing_with_nothing_supported_abstains_with_a_reason() -> None:
    nodes = make_nodes(ScriptedClient())
    state = state_with_context()
    generated = GeneratedAnswer.model_validate(BAD)
    state.generated = generated
    state.reports = [verify_claims(generated.claims, {"P1": state.context[0]})]

    answer = (await nodes.finalize(state))["answer"]

    assert answer.abstained is True  # type: ignore[attr-defined]
    assert answer.abstain_reason is AbstainReason.UNSUPPORTED_CITATIONS  # type: ignore[attr-defined]


async def test_abstaining_explains_why() -> None:
    nodes = make_nodes(ScriptedClient())
    with_context = state_with_context()
    empty = state_with_context()
    empty.context = []
    out_of_budget = state_with_context()
    out_of_budget.stop_reason = AbstainReason.BUDGET_EXCEEDED

    assert (await nodes.abstain(with_context))[
        "answer"
    ].abstain_reason is AbstainReason.INSUFFICIENT_EVIDENCE  # type: ignore[attr-defined]
    assert (await nodes.abstain(empty))[
        "answer"
    ].abstain_reason is AbstainReason.NO_RELEVANT_CONTEXT  # type: ignore[attr-defined]
    assert (await nodes.abstain(out_of_budget))[
        "answer"
    ].abstain_reason is AbstainReason.BUDGET_EXCEEDED  # type: ignore[attr-defined]
