import pytest

from grounded_rag.domain.principal import Principal
from grounded_rag.domain.retrieval import RetrievalMode
from grounded_rag.retrieval.dense import DenseRetriever
from grounded_rag.retrieval.service import (
    RetrievalConfig,
    RetrievalRequest,
    RetrievalService,
)
from tests.fakes import World

LEGAL = Principal("analyst", frozenset({"legal"}))
GOVERNING = "This agreement is governed by the laws of the State of Delaware."
PAYMENT = "Fees are payable within thirty days of the invoice date."
TERMINATION = "Either party may terminate for convenience on ninety days notice."


@pytest.fixture
async def world() -> World:
    world = World()
    await world.add_document("d1", "Supply Agreement", {"legal"}, [GOVERNING, PAYMENT, TERMINATION])
    return world


def service_for(world: World, **config: int) -> RetrievalService:
    return RetrievalService(
        DenseRetriever(world.embedder, world.vectors), world.chunks, RetrievalConfig(**config)
    )


def request(question: str, **overrides: object) -> RetrievalRequest:
    values: dict[str, object] = {"question": question, "principal": LEGAL, "corpus": "default"}
    return RetrievalRequest(**(values | overrides))  # type: ignore[arg-type]


async def test_the_most_relevant_chunk_is_ranked_first(world: World) -> None:
    result = await service_for(world).retrieve(request("which law governs the agreement"))
    assert result.chunks[0].chunk.text == GOVERNING


async def test_hybrid_trace_lists_every_stage_in_order(world: World) -> None:
    result = await service_for(world).retrieve(request("governing law"))
    assert [stage.name for stage in result.trace.stages] == ["dense", "lexical", "fused", "final"]


async def test_dense_mode_never_runs_the_lexical_search(world: World) -> None:
    result = await service_for(world).retrieve(request("governing law", mode=RetrievalMode.DENSE))
    assert [stage.name for stage in result.trace.stages] == ["dense", "fused", "final"]


async def test_lexical_mode_never_embeds_the_question(world: World) -> None:
    result = await service_for(world).retrieve(request("governing law", mode=RetrievalMode.LEXICAL))
    assert [stage.name for stage in result.trace.stages] == ["lexical", "fused", "final"]
    assert world.embedder.calls == []
    assert result.usage.embedding_tokens == 0


async def test_top_k_limits_the_final_chunks_only(world: World) -> None:
    result = await service_for(world).retrieve(request("agreement party days", top_k=2))
    assert len(result.chunks) == 2
    fused = next(s for s in result.trace.stages if s.name == "fused")
    assert len(fused.hits) == 3


async def test_candidate_limit_caps_the_fused_pool(world: World) -> None:
    result = await service_for(world, candidates=2).retrieve(request("agreement party days"))
    fused = next(s for s in result.trace.stages if s.name == "fused")
    assert len(fused.hits) == 2


async def test_question_embedding_cost_is_reported(world: World) -> None:
    result = await service_for(world).retrieve(request("one two three"))
    assert result.usage.embedding_tokens == 3


async def test_repeated_passages_are_collapsed_to_the_best_ranked_copy() -> None:
    world = World()
    await world.add_document(
        "d1", "Doc", {"legal"}, [GOVERNING, "unrelated filler", GOVERNING.upper()]
    )

    result = await service_for(world).retrieve(request("governed by Delaware law"))

    texts = [item.chunk.text.casefold() for item in result.chunks]
    assert texts.count(GOVERNING.casefold()) == 1


async def test_documents_that_are_not_ready_are_never_returned() -> None:
    world = World()
    await world.add_document("d1", "Draft", {"legal"}, [GOVERNING], status="pending")

    result = await service_for(world).retrieve(request("governing law Delaware"))

    assert result.chunks == []


async def test_a_principal_without_roles_gets_nothing_and_no_store_is_called(world: World) -> None:
    nobody = Principal("nobody", frozenset())

    result = await service_for(world).retrieve(request("governing law", principal=nobody))

    assert result.chunks == []
    assert world.embedder.calls == []
    assert world.chunks.lookups == []


async def test_other_corpora_are_not_searched() -> None:
    world = World()
    await world.add_document("d1", "Doc", {"legal"}, [GOVERNING], corpus="elsewhere")

    result = await service_for(world).retrieve(request("governing law Delaware"))

    assert result.chunks == []


class ReversingReranker:
    def __init__(self) -> None:
        self.received: list[list[str]] = []

    async def rerank(self, question: str, chunks):  # type: ignore[no-untyped-def]
        self.received.append([item.chunk.id for item in chunks])
        return list(reversed(range(len(chunks))))


async def test_reranker_decides_the_final_order(world: World) -> None:
    reranker = ReversingReranker()
    plain = await service_for(world).retrieve(request("agreement party days"))

    reranked = await service_for(world).retrieve(request("agreement party days"), reranker=reranker)

    assert [i.chunk.id for i in reranked.chunks] == [i.chunk.id for i in reversed(plain.chunks)]
    assert [s.name for s in reranked.trace.stages][-2:] == ["rerank", "final"]


async def test_top_k_is_applied_after_reranking(world: World) -> None:
    plain = await service_for(world).retrieve(request("agreement party days"))
    reranked = await service_for(world).retrieve(
        request("agreement party days", top_k=1), reranker=ReversingReranker()
    )

    assert [i.chunk.id for i in reranked.chunks] == [plain.chunks[-1].chunk.id]


async def test_the_reranker_only_ever_sees_chunks_the_principal_may_read() -> None:
    world = World()
    await world.add_document("ok", "Open", {"legal"}, [GOVERNING])
    await world.add_document("hidden", "Hidden", {"finance"}, [GOVERNING + " Secret."])
    reranker = ReversingReranker()

    await service_for(world).retrieve(request("governing law Delaware"), reranker=reranker)

    assert reranker.received == [["ok-00000"]]
