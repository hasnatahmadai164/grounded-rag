import pytest

from grounded_rag.agent.config import AgentConfig
from grounded_rag.agent.graph import build_graph, run_graph
from grounded_rag.agent.nodes import AgentNodes
from grounded_rag.agent.state import AgentState
from grounded_rag.domain.answer import AbstainReason
from grounded_rag.domain.principal import Principal
from grounded_rag.llm.budget import Budget, MeteredLLM, UsageMeter
from grounded_rag.retrieval.dense import DenseRetriever
from grounded_rag.retrieval.rerank import LLMReranker
from grounded_rag.retrieval.rewrite import QueryRewriter
from grounded_rag.retrieval.service import RetrievalService
from tests.fakes import PurposeClient, World

LEGAL = Principal("analyst", frozenset({"legal"}))
GOVERNING = "This Agreement shall be governed by the laws of the State of Delaware."
PAYMENT = "Fees are payable within thirty (30) days of the invoice date."
QUESTION = "Which law governs the Supply Agreement?"

SUFFICIENT = {"sufficient": True, "reason": "Clause found.", "missing": ""}
INSUFFICIENT = {
    "sufficient": False,
    "reason": "No governing law clause.",
    "missing": "governing law",
}
GOOD_ANSWER = {
    "answerable": True,
    "claims": [
        {
            "text": "Delaware law governs.",
            "citations": [{"passage_id": "P1", "quote": "the laws of the State of Delaware"}],
        }
    ],
}
BAD_ANSWER = {
    "answerable": True,
    "claims": [
        {
            "text": "Delaware law governs.",
            "citations": [{"passage_id": "P1", "quote": "the laws of the State of California"}],
        }
    ],
}


async def make_world() -> World:
    world = World()
    await world.add_document("d1", "Supply Agreement", {"legal"}, [GOVERNING, PAYMENT])
    await world.add_document(
        "d2", "Finance Memo", {"finance"}, ["Secret revenue share of twelve percent."]
    )
    return world


async def run(
    client: PurposeClient, *, budget: Budget | None = None, world: World | None = None, **cfg
):
    world = world or await make_world()
    config = AgentConfig(**{"rerank": False} | cfg)  # type: ignore[arg-type]
    meter = UsageMeter()
    llm = MeteredLLM(client, meter, budget or Budget(1_000_000, 100.0))
    nodes = AgentNodes(
        retrieval=RetrievalService(DenseRetriever(world.embedder, world.vectors), world.chunks),
        llm=llm,
        meter=meter,
        reranker=LLMReranker(llm) if config.rerank else None,
        rewriter=QueryRewriter(llm),
        config=config,
    )
    state = AgentState(QUESTION, LEGAL, "default", config.top_k, [QUESTION])
    final = await run_graph(build_graph(nodes, config), state)
    return final, meter


async def test_a_well_supported_question_gets_a_cited_answer() -> None:
    client = PurposeClient(grade=[SUFFICIENT], generate=[GOOD_ANSWER])

    final, meter = await run(client)

    assert final.answer is not None
    assert final.answer.abstained is False
    assert final.answer.text == "Delaware law governs. [1]"
    assert final.answer.citations[0].quote == "the laws of the State of Delaware"
    assert client.purposes == ["grade", "generate"]
    assert meter.calls == 2


async def test_the_whole_path_is_recorded_as_events() -> None:
    final, _ = await run(PurposeClient(grade=[SUFFICIENT], generate=[GOOD_ANSWER]))
    assert [event.node for event in final.events] == [
        "retrieve",
        "grade",
        "generate",
        "verify",
        "finalize",
    ]


async def test_insufficient_context_triggers_a_rewrite_and_a_second_retrieval() -> None:
    client = PurposeClient(
        grade=[INSUFFICIENT, SUFFICIENT],
        rewrite=[{"query": "governed by laws of the State of Delaware"}],
        generate=[GOOD_ANSWER],
    )

    final, _ = await run(client)

    assert client.purposes == ["grade", "rewrite", "grade", "generate"]
    assert final.queries == [QUESTION, "governed by laws of the State of Delaware"]
    assert len(final.retrievals) == 2
    assert final.rewrites_used == 1
    assert final.answer is not None and not final.answer.abstained


async def test_rewrites_are_bounded_and_the_loop_then_abstains() -> None:
    client = PurposeClient(
        grade=[INSUFFICIENT] * 3,
        rewrite=[{"query": "first rewrite"}, {"query": "second rewrite"}],
    )

    final, _ = await run(client, max_rewrites=2)

    assert final.rewrites_used == 2
    assert len(final.retrievals) == 3
    assert final.answer is not None
    assert final.answer.abstain_reason is AbstainReason.INSUFFICIENT_EVIDENCE
    assert "generate" not in client.purposes


async def test_no_retry_configuration_abstains_after_one_insufficient_grade() -> None:
    client = PurposeClient(grade=[INSUFFICIENT])

    final, _ = await run(client, max_rewrites=0)

    assert client.purposes == ["grade"]
    assert final.answer is not None and final.answer.abstained


async def test_a_rewrite_that_repeats_an_earlier_query_stops_the_loop() -> None:
    client = PurposeClient(
        grade=[INSUFFICIENT], rewrite=[{"query": "  WHICH law governs the supply agreement? "}]
    )

    final, _ = await run(client, max_rewrites=2)

    assert final.stalled is True
    assert len(final.retrievals) == 1
    assert final.answer is not None and final.answer.abstained


async def test_a_bad_quote_is_repaired_once() -> None:
    client = PurposeClient(grade=[SUFFICIENT], generate=[BAD_ANSWER], repair=[GOOD_ANSWER])

    final, _ = await run(client)

    assert client.purposes == ["grade", "generate", "repair"]
    assert final.repairs_used == 1
    assert [r.all_verified for r in final.reports] == [False, True]
    assert final.answer is not None and final.answer.text == "Delaware law governs. [1]"


async def test_a_second_failure_drops_the_claim_and_abstains() -> None:
    client = PurposeClient(grade=[SUFFICIENT], generate=[BAD_ANSWER], repair=[BAD_ANSWER])

    final, _ = await run(client)

    assert final.repairs_used == 1
    assert final.answer is not None
    assert final.answer.abstain_reason is AbstainReason.UNSUPPORTED_CITATIONS
    assert client.purposes.count("repair") == 1


async def test_a_model_that_says_it_cannot_answer_leads_to_abstention() -> None:
    client = PurposeClient(grade=[SUFFICIENT], generate=[{"answerable": False, "claims": []}])

    final, _ = await run(client)

    assert final.answer is not None
    assert final.answer.abstain_reason is AbstainReason.INSUFFICIENT_EVIDENCE
    assert "verify" not in [e.node for e in final.events]


async def test_an_empty_corpus_abstains_without_any_model_call() -> None:
    client = PurposeClient()

    final, meter = await run(client, world=World())

    assert final.answer is not None
    assert final.answer.abstain_reason is AbstainReason.NO_RELEVANT_CONTEXT
    assert meter.calls == 0


async def test_exhausting_the_budget_ends_in_a_budget_abstention() -> None:
    client = PurposeClient(grade=[SUFFICIENT])

    final, _ = await run(client, budget=Budget(max_tokens=200, max_cost_usd=100.0))

    assert final.answer is not None
    assert final.answer.abstain_reason is AbstainReason.BUDGET_EXCEEDED
    assert client.purposes == ["grade"]


async def test_reranking_adds_one_model_call_before_grading() -> None:
    client = PurposeClient(rerank=[{"ranking": [1, 2]}], grade=[SUFFICIENT], generate=[GOOD_ANSWER])

    final, _ = await run(client, rerank=True)

    assert client.purposes == ["rerank", "grade", "generate"]
    assert final.answer is not None and not final.answer.abstained


async def test_the_model_never_receives_text_from_documents_the_principal_cannot_read() -> None:
    client = PurposeClient(grade=[SUFFICIENT], generate=[GOOD_ANSWER])

    await run(client)

    everything_sent = " ".join(m.content for r in client.requests for m in r.messages)
    assert "Secret revenue share" not in everything_sent
    assert "Finance Memo" not in everything_sent


@pytest.mark.parametrize("max_rewrites", [0, 1, 2, 3])
async def test_retrieval_count_never_exceeds_rewrites_plus_one(max_rewrites: int) -> None:
    client = PurposeClient(
        grade=[INSUFFICIENT] * (max_rewrites + 1),
        rewrite=[{"query": f"variation number {n}"} for n in range(max_rewrites)],
    )

    final, _ = await run(client, max_rewrites=max_rewrites)

    assert len(final.retrievals) == max_rewrites + 1
