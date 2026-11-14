from dataclasses import dataclass

from grounded_rag.domain.retrieval import Hit
from grounded_rag.domain.usage import Usage
from grounded_rag.ingestion.embedder import Embedder
from grounded_rag.vectorstore.base import VectorStore


@dataclass(frozen=True, slots=True)
class DenseResult:
    hits: list[Hit]
    usage: Usage
    embedding_latency_ms: float
    replay_credit_ms: float = 0.0


class DenseRetriever:
    """Embeds the question and asks the vector store for the nearest chunks the roles may read."""

    def __init__(self, embedder: Embedder, vectors: VectorStore) -> None:
        self._embedder = embedder
        self._vectors = vectors

    async def search(
        self, corpus: str, roles: frozenset[str], question: str, *, limit: int
    ) -> DenseResult:
        embedded = await self._embedder.embed([question])
        matches = await self._vectors.query(corpus, embedded.vectors[0], top_k=limit, roles=roles)
        hits = [Hit(chunk_id=match.id, score=match.score) for match in matches]
        return DenseResult(hits, embedded.usage, embedded.latency_ms, embedded.replay_credit_ms)
