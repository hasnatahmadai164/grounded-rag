import pytest

from grounded_rag.agent.config import AgentConfig
from grounded_rag.agent.service import AnswerService
from grounded_rag.domain.answer import AbstainReason
from grounded_rag.domain.principal import Principal
from grounded_rag.llm.budget import Budget
from grounded_rag.retrieval.dense import DenseRetriever
from grounded_rag.retrieval.service import RetrievalService
from tests.fakes import PurposeClient, World

LEGAL = Principal("analyst", frozenset({"legal"}))
GOVERNING = "This Agreement shall be governed by the laws of the State of Delaware."
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


@pytest.fixture
async def world() -> World:
    world = World()
    await world.add_document(
        "d1", "Supply Agreement", {"legal"}, [GOVERNING, "Fees are due monthly."]
    )
    return world


def service_for(world: World, client: PurposeClient, **config: object) -> AnswerService:
    retrieval = RetrievalService(DenseRetriever(world.embedder, world.vectors), world.chunks)
    return AnswerService(
        retrieval,
        client,
        Budget(1_000_000, 100.0),
        AgentConfig(**({"rerank": False} | config)),  # type: ignore[arg-type]
    )


async def test_the_result_carries_the_answer_usage_and_trace(world: World) -> None:
    client = PurposeClient(grade=[SUFFICIENT], generate=[ANSWER])

    result = await service_for(world, client).answer("Which law governs?", LEGAL, corpus="default")

    assert result.answer.text == "Delaware law governs. [1]"
    assert result.llm_calls == 2
    assert set(result.by_purpose) == {"embedding", "grade", "generate"}
    assert result.usage.prompt_tokens == 400
    assert result.usage.completion_tokens == 80
    assert result.usage.embedding_tokens == 3
    assert result.usage.cost_usd == pytest.approx(0.0002)
    assert result.latency_ms > 0
    assert [e.node for e in result.state.events][-1] == "finalize"


async def test_each_call_gets_its_own_meter(world: World) -> None:
    client = PurposeClient(grade=[SUFFICIENT, SUFFICIENT], generate=[ANSWER, ANSWER])
    service = service_for(world, client)

    first = await service.answer("Which law governs?", LEGAL, corpus="default")
    second = await service.answer("Which law governs?", LEGAL, corpus="default")

    assert first.usage == second.usage


async def test_a_per_call_config_replaces_the_defaults(world: World) -> None:
    client = PurposeClient(grade=[{"sufficient": False, "reason": "no", "missing": "x"}])
    service = service_for(world, client, max_rewrites=2)

    result = await service.answer(
        "Which law governs?",
        LEGAL,
        corpus="default",
        config=AgentConfig(rerank=False, max_rewrites=0),
    )

    assert result.answer.abstain_reason is AbstainReason.INSUFFICIENT_EVIDENCE
    assert client.purposes == ["grade"]
    assert result.config.max_rewrites == 0


async def test_top_k_overrides_the_configured_value(world: World) -> None:
    client = PurposeClient(grade=[SUFFICIENT], generate=[ANSWER])

    result = await service_for(world, client).answer(
        "Which law governs the agreement fees?", LEGAL, corpus="default", top_k=1
    )

    assert len(result.state.retrievals[0].chunks) == 1


async def test_reported_latency_adds_back_time_saved_by_a_cache(world: World) -> None:
    client = PurposeClient(grade=[SUFFICIENT], generate=[ANSWER])

    result = await service_for(world, client).answer("Which law governs?", LEGAL, corpus="default")

    assert result.reported_latency_ms == pytest.approx(result.latency_ms + result.replay_credit_ms)
