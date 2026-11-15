import pytest

from grounded_rag.domain.principal import Principal
from grounded_rag.domain.retrieval import RetrievalMode, RetrievalResult
from grounded_rag.retrieval.dense import DenseRetriever
from grounded_rag.retrieval.service import RetrievalRequest, RetrievalService
from tests.fakes import World

LEGAL = Principal("legal-analyst", frozenset({"legal"}))
FINANCE = Principal("finance-analyst", frozenset({"finance"}))
SECRET_TEXT = "The confidential revenue share is twelve percent of net sales."
OPEN_TEXT = "The governing law is Delaware."
QUESTION = "what is the revenue share and governing law"
MODES = list(RetrievalMode)


async def build_world() -> World:
    world = World()
    await world.add_document("fin", "Finance Memo", {"finance"}, [SECRET_TEXT])
    await world.add_document("leg", "Legal Memo", {"legal"}, [OPEN_TEXT])
    return world


def service(world: World) -> RetrievalService:
    return RetrievalService(DenseRetriever(world.embedder, world.vectors), world.chunks)


async def retrieve(world: World, principal: Principal, mode: RetrievalMode) -> RetrievalResult:
    request = RetrievalRequest(QUESTION, principal, "default", mode=mode, top_k=10)
    return await service(world).retrieve(request)


def every_chunk_id(result: RetrievalResult) -> set[str]:
    ids = {item.chunk.id for item in result.chunks}
    for stage in result.trace.stages:
        ids.update(hit.chunk_id for hit in stage.hits)
    ids.update(result.candidates)
    return ids


@pytest.mark.parametrize("mode", MODES)
async def test_a_principal_never_sees_chunks_of_documents_it_cannot_read(
    mode: RetrievalMode,
) -> None:
    world = await build_world()

    result = await retrieve(world, LEGAL, mode)

    assert every_chunk_id(result) == {"leg-00000"}
    assert all(SECRET_TEXT not in item.chunk.text for item in result.chunks)


@pytest.mark.parametrize("mode", MODES)
async def test_stale_vector_metadata_cannot_leak_a_chunk(mode: RetrievalMode) -> None:
    world = World()
    await world.add_document(
        "fin", "Finance Memo", {"finance"}, [SECRET_TEXT], vector_roles={"legal"}
    )

    result = await retrieve(world, LEGAL, mode)

    assert every_chunk_id(result) == set()
    assert result.dropped_unauthorized >= 0


async def test_narrowed_roles_apply_even_though_vectors_still_carry_the_old_ones() -> None:
    world = World()
    await world.add_document("memo", "Memo", {"legal"}, [OPEN_TEXT])
    assert (await retrieve(world, LEGAL, RetrievalMode.HYBRID)).chunks

    for row in world.chunks.rows:
        row["roles"] = frozenset({"finance"})

    assert (await retrieve(world, LEGAL, RetrievalMode.HYBRID)).chunks == []


@pytest.mark.parametrize("mode", MODES)
async def test_a_lexical_search_that_ignores_roles_is_still_filtered(mode: RetrievalMode) -> None:
    world = await build_world()
    world.chunks.ignore_roles_in_search = True

    result = await retrieve(world, LEGAL, mode)

    assert every_chunk_id(result) == {"leg-00000"}


@pytest.mark.parametrize("mode", MODES)
async def test_a_lookup_that_ignores_roles_is_still_filtered(mode: RetrievalMode) -> None:
    world = await build_world()
    world.chunks.ignore_roles_in_search = True
    world.chunks.ignore_roles_in_lookup = True

    result = await retrieve(world, LEGAL, mode)

    assert every_chunk_id(result) == {"leg-00000"}


async def test_the_vector_store_is_asked_for_the_principals_roles_only() -> None:
    world = await build_world()

    await retrieve(world, LEGAL, RetrievalMode.DENSE)

    assert world.vectors.queries == [frozenset({"legal"})]


@pytest.mark.parametrize("mode", MODES)
async def test_a_forbidden_document_gives_the_same_result_as_a_missing_one(
    mode: RetrievalMode,
) -> None:
    with_secret = await build_world()
    without_secret = World()
    await without_secret.add_document("leg", "Legal Memo", {"legal"}, [OPEN_TEXT])

    present = await retrieve(with_secret, LEGAL, mode)
    absent = await retrieve(without_secret, LEGAL, mode)

    assert [i.chunk.id for i in present.chunks] == [i.chunk.id for i in absent.chunks]
    assert [(h.chunk_id, h.score) for h in present.hits] == [
        (h.chunk_id, h.score) for h in absent.hits
    ]
    for shown, hidden in zip(present.trace.stages, absent.trace.stages, strict=True):
        assert shown.name == hidden.name
        assert [h.chunk_id for h in shown.hits] == [h.chunk_id for h in hidden.hits]


async def test_different_principals_see_different_documents_for_the_same_question() -> None:
    world = await build_world()

    legal = await retrieve(world, LEGAL, RetrievalMode.HYBRID)
    finance = await retrieve(world, FINANCE, RetrievalMode.HYBRID)

    assert every_chunk_id(legal) == {"leg-00000"}
    assert every_chunk_id(finance) == {"fin-00000"}


async def test_a_principal_holding_several_roles_sees_all_matching_documents() -> None:
    world = await build_world()
    both = Principal("auditor", frozenset({"legal", "finance"}))

    result = await retrieve(world, both, RetrievalMode.HYBRID)

    assert every_chunk_id(result) == {"leg-00000", "fin-00000"}


async def test_a_principal_without_roles_receives_nothing_from_any_mode() -> None:
    world = await build_world()
    nobody = Principal("nobody", frozenset())

    for mode in MODES:
        result = await retrieve(world, nobody, mode)
        assert every_chunk_id(result) == set()


@pytest.mark.parametrize("status", ["pending", "deleting"])
async def test_documents_that_are_not_ready_stay_hidden_even_if_vectors_remain(status: str) -> None:
    world = World()
    await world.add_document("fin", "Finance Memo", {"legal"}, [SECRET_TEXT], status=status)

    result = await retrieve(world, LEGAL, RetrievalMode.HYBRID)

    assert every_chunk_id(result) == set()
