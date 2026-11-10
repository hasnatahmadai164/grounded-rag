from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """A chunk embedding plus the only metadata the vector store may hold.

    Text never leaves PostgreSQL. `allowed_roles` is a pre-filter for recall; the
    authoritative access check happens against PostgreSQL after retrieval.
    """

    id: str
    values: Sequence[float]
    doc_id: str
    allowed_roles: frozenset[str]
    section: str | None = None


@dataclass(frozen=True, slots=True)
class VectorMatch:
    id: str
    score: float


class VectorStore(Protocol):
    """One namespace per corpus. Implementations must make `delete` safe to repeat."""

    async def upsert(self, corpus: str, records: Sequence[VectorRecord]) -> None: ...

    async def query(
        self, corpus: str, vector: Sequence[float], *, top_k: int, roles: frozenset[str]
    ) -> list[VectorMatch]:
        """Nearest neighbours among vectors whose `allowed_roles` intersect `roles`."""
        ...

    async def delete(self, corpus: str, ids: Sequence[str]) -> None: ...

    async def count(self, corpus: str) -> int:
        """Number of vectors currently visible in the corpus. Doubles as a reachability check."""
        ...
