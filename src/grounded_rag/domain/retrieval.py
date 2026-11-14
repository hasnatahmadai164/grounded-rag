from dataclasses import dataclass, field
from enum import StrEnum

from grounded_rag.domain.documents import RetrievedChunk
from grounded_rag.domain.usage import Usage


class RetrievalMode(StrEnum):
    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class Hit:
    """A chunk id with the score one retrieval stage gave it."""

    chunk_id: str
    score: float


@dataclass(frozen=True, slots=True)
class StageTrace:
    """Candidates one stage produced, restricted to chunks the caller is allowed to read."""

    name: str
    hits: tuple[Hit, ...]
    duration_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    stages: tuple[StageTrace, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Final chunks in rank order plus what is needed to explain how they were chosen.

    `candidates` holds every authorised chunk any stage returned, for presenting the trace.
    `dropped_unauthorized` is for logs only and must not be shown to callers.
    """

    chunks: list[RetrievedChunk]
    hits: list[Hit]
    trace: RetrievalTrace
    usage: Usage
    candidates: dict[str, RetrievedChunk] = field(default_factory=dict)
    dropped_unauthorized: int = 0
    replay_credit_ms: float = 0.0
