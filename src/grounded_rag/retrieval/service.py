import asyncio
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from grounded_rag.domain.documents import RetrievedChunk
from grounded_rag.domain.principal import Principal
from grounded_rag.domain.retrieval import (
    Hit,
    RetrievalMode,
    RetrievalResult,
    RetrievalTrace,
    StageTrace,
)
from grounded_rag.domain.usage import Usage
from grounded_rag.logging import get_logger
from grounded_rag.retrieval.access import ChunkLookup, authorize_hits
from grounded_rag.retrieval.dense import DenseRetriever
from grounded_rag.retrieval.fusion import reciprocal_rank_fusion
from grounded_rag.retrieval.lexical import LexicalIndex, lexical_search
from grounded_rag.retrieval.rerank import Reranker

log = get_logger(__name__)

_WHITESPACE = re.compile(r"\s+")


class ChunkStore(LexicalIndex, ChunkLookup, Protocol):
    """What retrieval needs from PostgreSQL: full-text search and the authoritative lookup."""


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    dense_top_k: int = 30
    lexical_top_k: int = 30
    rrf_k: int = 60
    candidates: int = 20


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    question: str
    principal: Principal
    corpus: str
    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = 6


class StageTimer:
    def __init__(self) -> None:
        self._started = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000


def drop_duplicate_texts(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the best-ranked copy of passages that repeat verbatim within a document."""
    seen: set[tuple[str, str]] = set()
    kept: list[RetrievedChunk] = []
    for item in chunks:
        key = (item.chunk.doc_id, _WHITESPACE.sub(" ", item.chunk.text).strip().casefold())
        if key not in seen:
            seen.add(key)
            kept.append(item)
    return kept


class RetrievalService:
    """Dense and lexical retrieval, fusion, and the final access check.

    Everything returned has passed the PostgreSQL access check, including the stage traces.
    """

    def __init__(
        self, dense: DenseRetriever, chunks: ChunkStore, config: RetrievalConfig | None = None
    ) -> None:
        self._dense = dense
        self._chunks = chunks
        self._config = config or RetrievalConfig()

    async def retrieve(
        self, request: RetrievalRequest, *, reranker: Reranker | None = None
    ) -> RetrievalResult:
        if not request.principal.roles:
            return RetrievalResult([], [], RetrievalTrace(), Usage())

        raw, usage, credit = await self._candidates(request)
        fused = self._fuse(request.mode, raw)

        timer = StageTimer()
        every_hit = {hit.chunk_id: hit for stage in raw for hit in stage.hits}
        authorized = await authorize_hits(
            self._chunks, request.principal, request.corpus, list(every_hit.values())
        )
        allowed = {item.chunk.id: item for item in authorized.chunks}
        if authorized.dropped:
            log.warning("unauthorized_candidates_dropped", count=authorized.dropped)

        stages = [
            StageTrace(
                stage.name, tuple(h for h in stage.hits if h.chunk_id in allowed), stage.duration_ms
            )
            for stage in raw
        ]
        ranked = [hit for hit in fused if hit.chunk_id in allowed][: self._config.candidates]
        stages.append(StageTrace("fused", tuple(ranked), timer.elapsed_ms()))

        unique = drop_duplicate_texts([allowed[hit.chunk_id] for hit in ranked])
        if reranker is not None:
            timer = StageTimer()
            unique = await self._rerank(reranker, request.question, unique)
            stages.append(StageTrace("rerank", self._positional_hits(unique), timer.elapsed_ms()))
        final = unique[: request.top_k]
        by_id = {hit.chunk_id: hit for hit in ranked}
        final_hits = [by_id[item.chunk.id] for item in final]
        stages.append(StageTrace("final", tuple(final_hits), 0.0))

        return RetrievalResult(
            chunks=final,
            hits=final_hits,
            trace=RetrievalTrace(tuple(stages)),
            usage=usage,
            candidates=allowed,
            dropped_unauthorized=authorized.dropped,
            replay_credit_ms=credit,
        )

    async def _candidates(self, request: RetrievalRequest) -> tuple[list[StageTrace], Usage, float]:
        roles = request.principal.roles
        wants_dense = request.mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID)
        wants_lexical = request.mode in (RetrievalMode.LEXICAL, RetrievalMode.HYBRID)

        async def run_dense() -> tuple[StageTrace, Usage, float] | None:
            if not wants_dense:
                return None
            timer = StageTimer()
            result = await self._dense.search(
                request.corpus, roles, request.question, limit=self._config.dense_top_k
            )
            stage = StageTrace("dense", tuple(result.hits), timer.elapsed_ms())
            return stage, result.usage, result.replay_credit_ms

        async def run_lexical() -> StageTrace | None:
            if not wants_lexical:
                return None
            timer = StageTimer()
            hits = await lexical_search(
                self._chunks,
                request.corpus,
                roles,
                request.question,
                limit=self._config.lexical_top_k,
            )
            return StageTrace("lexical", tuple(hits), timer.elapsed_ms())

        dense, lexical = await asyncio.gather(run_dense(), run_lexical())
        stages: list[StageTrace] = []
        usage, credit = Usage(), 0.0
        if dense is not None:
            stages.append(dense[0])
            usage, credit = dense[1], dense[2]
        if lexical is not None:
            stages.append(lexical)
        return stages, usage, credit

    @staticmethod
    async def _rerank(
        reranker: Reranker, question: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        order = await reranker.rerank(question, chunks)
        return [chunks[index] for index in order]

    @staticmethod
    def _positional_hits(chunks: Sequence[RetrievedChunk]) -> tuple[Hit, ...]:
        """Rerankers give an order, not scores, so scores fall linearly from 1 to 1/n."""
        size = len(chunks)
        return tuple(
            Hit(item.chunk.id, (size - position) / size) for position, item in enumerate(chunks)
        )

    def _fuse(self, mode: RetrievalMode, raw: list[StageTrace]) -> list[Hit]:
        if mode is RetrievalMode.HYBRID:
            return reciprocal_rank_fusion([stage.hits for stage in raw], k=self._config.rrf_k)
        return list(raw[0].hits)
