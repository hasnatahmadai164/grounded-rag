from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from grounded_rag.domain.documents import RetrievedChunk
from grounded_rag.domain.principal import Principal
from grounded_rag.domain.retrieval import Hit
from grounded_rag.logging import get_logger

log = get_logger(__name__)


class ChunkLookup(Protocol):
    async def visible_chunks(
        self, corpus: str, roles: frozenset[str], chunk_ids: Sequence[str]
    ) -> list[RetrievedChunk]: ...


@dataclass(frozen=True, slots=True)
class AuthorizedHits:
    """Hits that passed the final access check, with their chunks, in the order given.

    `dropped` counts hits that failed. It is for operators and must never reach a caller:
    it would reveal that a document the caller cannot see matched the question.
    """

    hits: list[Hit]
    chunks: list[RetrievedChunk]
    dropped: int


async def authorize_hits(
    lookup: ChunkLookup, principal: Principal, corpus: str, hits: Sequence[Hit]
) -> AuthorizedHits:
    """The third access check, and the one that counts.

    Vector metadata can be stale and the first two filters are optimisations, so every
    candidate is looked up again in PostgreSQL. The lookup filters in SQL; the roles are
    then compared once more here so a faulty lookup cannot widen access.
    """
    if not hits or not principal.roles:
        return AuthorizedHits([], [], dropped=len(hits))

    requested = {hit.chunk_id for hit in hits}
    found = await lookup.visible_chunks(corpus, principal.roles, [h.chunk_id for h in hits])
    by_id: dict[str, RetrievedChunk] = {}
    for item in found:
        if item.chunk.id not in requested:
            continue
        if not principal.can_see(item.allowed_roles):
            log.error("authorization_recheck_rejected_chunk", chunk_id=item.chunk.id)
            continue
        by_id[item.chunk.id] = item

    kept = [hit for hit in hits if hit.chunk_id in by_id]
    return AuthorizedHits(
        hits=kept,
        chunks=[by_id[hit.chunk_id] for hit in kept],
        dropped=len(hits) - len(kept),
    )
