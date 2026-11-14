from collections.abc import Sequence

from grounded_rag.domain.documents import Chunk, RetrievedChunk
from grounded_rag.domain.principal import Principal
from grounded_rag.domain.retrieval import Hit
from grounded_rag.retrieval.access import authorize_hits

LEGAL = Principal("analyst", frozenset({"legal"}))


def retrieved(chunk_id: str, roles: set[str]) -> RetrievedChunk:
    chunk = Chunk(chunk_id, "doc", 0, f"text of {chunk_id}", 0, 10, None, "Doc")
    return RetrievedChunk(chunk=chunk, document_title="Doc", allowed_roles=frozenset(roles))


def hits(*ids: str) -> list[Hit]:
    return [Hit(i, 1.0) for i in ids]


class StaticLookup:
    """Returns exactly what it is given, ignoring the roles: a worst-case faulty lookup."""

    def __init__(self, *items: RetrievedChunk) -> None:
        self.items = list(items)
        self.calls = 0

    async def visible_chunks(
        self, corpus: str, roles: frozenset[str], chunk_ids: Sequence[str]
    ) -> list[RetrievedChunk]:
        self.calls += 1
        return self.items


async def test_chunks_missing_from_the_database_answer_are_dropped() -> None:
    lookup = StaticLookup(retrieved("a", {"legal"}))

    result = await authorize_hits(lookup, LEGAL, "c", hits("a", "stale"))

    assert [h.chunk_id for h in result.hits] == ["a"]
    assert result.dropped == 1


async def test_original_ranking_order_is_preserved() -> None:
    lookup = StaticLookup(
        retrieved("c", {"legal"}), retrieved("a", {"legal"}), retrieved("b", {"legal"})
    )

    result = await authorize_hits(lookup, LEGAL, "c", hits("b", "c", "a"))

    assert [c.chunk.id for c in result.chunks] == ["b", "c", "a"]


async def test_a_lookup_that_returns_a_forbidden_chunk_cannot_widen_access() -> None:
    lookup = StaticLookup(retrieved("ok", {"legal"}), retrieved("forbidden", {"finance"}))

    result = await authorize_hits(lookup, LEGAL, "c", hits("ok", "forbidden"))

    assert [c.chunk.id for c in result.chunks] == ["ok"]


async def test_chunks_that_were_never_requested_are_ignored() -> None:
    lookup = StaticLookup(retrieved("asked", {"legal"}), retrieved("extra", {"legal"}))

    result = await authorize_hits(lookup, LEGAL, "c", hits("asked"))

    assert [c.chunk.id for c in result.chunks] == ["asked"]


async def test_principal_without_roles_is_denied_without_a_lookup() -> None:
    lookup = StaticLookup(retrieved("a", {"legal"}))

    result = await authorize_hits(lookup, Principal("nobody", frozenset()), "c", hits("a"))

    assert result.chunks == []
    assert lookup.calls == 0


async def test_no_hits_means_no_lookup() -> None:
    lookup = StaticLookup()
    result = await authorize_hits(lookup, LEGAL, "c", [])
    assert result.chunks == []
    assert result.dropped == 0
    assert lookup.calls == 0
