import json

from grounded_rag.agent.config import AgentConfig
from grounded_rag.agent.nodes import AgentNodes, merge_context
from grounded_rag.agent.state import AgentState
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

LEGAL = Principal("analyst", frozenset({"legal"}))
GOVERNING = "This agreement is governed by the laws of the State of Delaware."


class ScriptedClient:
    def __init__(self, *responses: dict | str) -> None:
        self.responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.requests: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        usage = Usage(prompt_tokens=100, completion_tokens=20, cost_usd=0.0001)
        return ChatResult(self.responses.pop(0), usage, 5.0, "m")


def retrieved(chunk_id: str, text: str = "text") -> RetrievedChunk:
    chunk = Chunk(chunk_id, "doc", 0, text, 0, len(text), None, "Doc")
    return RetrievedChunk(chunk, "Doc", frozenset({"legal"}))


async def build(client: ScriptedClient, *, budget: Budget | None = None, **config: object):
    world = World()
    await world.add_document(
        "d1", "Supply Agreement", {"legal"}, [GOVERNING, "Payment in thirty days."]
    )
    meter = UsageMeter()
    llm = MeteredLLM(client, meter, budget or Budget(100_000, 10.0))
    nodes = AgentNodes(
        retrieval=RetrievalService(DenseRetriever(world.embedder, world.vectors), world.chunks),
        llm=llm,
        meter=meter,
        reranker=None,
        rewriter=QueryRewriter(llm),
        config=AgentConfig(**config),  # type: ignore[arg-type]
    )
    return nodes, meter


def state_for(question: str = "which law governs the agreement") -> AgentState:
    return AgentState(
        question=question, principal=LEGAL, corpus="default", top_k=6, queries=[question]
    )


def test_new_results_come_first_and_duplicates_are_removed() -> None:
    merged = merge_context([retrieved("b"), retrieved("a")], [retrieved("a"), retrieved("c")], 10)
    assert [m.chunk.id for m in merged] == ["b", "a", "c"]


def test_merged_context_is_capped() -> None:
    merged = merge_context([retrieved("a")], [retrieved("b"), retrieved("c")], 2)
    assert [m.chunk.id for m in merged] == ["a", "b"]


async def test_retrieve_fills_the_context_and_records_embedding_cost() -> None:
    nodes, meter = await build(ScriptedClient())

    update = await nodes.retrieve(state_for())

    assert next(c.chunk.id for c in update["context"]) == "d1-00000"  # type: ignore[union-attr]
    assert meter.usage.embedding_tokens > 0
    assert update["events"][0].node == "retrieve"  # type: ignore[index]


async def test_a_rewritten_query_adds_to_what_was_already_found() -> None:
    nodes, _ = await build(ScriptedClient())
    state = state_for()
    state.context = [retrieved("earlier")]
    state.queries.append("law of Delaware")

    update = await nodes.retrieve(state)

    ids = [c.chunk.id for c in update["context"]]  # type: ignore[union-attr]
    assert "earlier" in ids
    assert ids[0] != "earlier"


async def test_grader_sees_the_original_question_and_labelled_passages() -> None:
    client = ScriptedClient({"sufficient": True, "reason": "stated", "missing": ""})
    nodes, meter = await build(client)
    state = state_for()
    state.queries.append("a rewritten query")
    state.context = [retrieved("c1", GOVERNING)]

    update = await nodes.grade(state)

    user_message = client.requests[0].messages[1].content
    assert "which law governs the agreement" in user_message
    assert "a rewritten query" not in user_message
    assert 'id="P1"' in user_message
    assert update["grade"].sufficient is True  # type: ignore[union-attr]
    assert meter.by_purpose["grade"].prompt_tokens == 100


async def test_grading_with_no_passages_needs_no_model_call() -> None:
    client = ScriptedClient()
    nodes, _ = await build(client)

    update = await nodes.grade(state_for())

    assert update["grade"].sufficient is False  # type: ignore[union-attr]
    assert client.requests == []


async def test_an_exhausted_budget_stops_the_loop_instead_of_failing() -> None:
    client = ScriptedClient({"sufficient": True, "reason": "x", "missing": ""})
    nodes, meter = await build(client, budget=Budget(10, 10.0))
    meter.record_embedding(Usage(embedding_tokens=50))
    state = state_for()
    state.context = [retrieved("c1")]

    update = await nodes.grade(state)

    assert update["stop_reason"] is AbstainReason.BUDGET_EXCEEDED
    assert client.requests == []
