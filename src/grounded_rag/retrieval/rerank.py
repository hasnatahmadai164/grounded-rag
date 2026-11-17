from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from grounded_rag.domain.documents import RetrievedChunk
from grounded_rag.llm.budget import MeteredLLM
from grounded_rag.llm.prompts import rerank_messages


class Reranker(Protocol):
    async def rerank(self, question: str, chunks: Sequence[RetrievedChunk]) -> list[int]:
        """Indices into `chunks`, most useful first. Every index appears exactly once."""
        ...


class RerankOutput(BaseModel):
    ranking: list[int] = Field(description="Passage numbers, most useful first.")


def complete_ranking(raw: Sequence[int], size: int) -> list[int]:
    """Turn the model's 1-based list into a full 0-based permutation.

    Out-of-range and repeated numbers are ignored; passages the model left out keep their
    original relative order at the end, so a sloppy answer never loses a candidate.
    """
    order: list[int] = []
    for number in raw:
        index = number - 1
        if 0 <= index < size and index not in order:
            order.append(index)
    order.extend(index for index in range(size) if index not in order)
    return order


class LLMReranker:
    """Listwise reranking with one model call over the candidates' opening text."""

    def __init__(self, llm: MeteredLLM) -> None:
        self._llm = llm

    async def rerank(self, question: str, chunks: Sequence[RetrievedChunk]) -> list[int]:
        if len(chunks) < 2:
            return list(range(len(chunks)))
        output = await self._llm.structured(
            rerank_messages(question, chunks), RerankOutput, purpose="rerank", max_output_tokens=256
        )
        return complete_ranking(output.ranking, len(chunks))


class KeepOrderReranker:
    """Leaves the fused order unchanged. Useful as an explicit baseline."""

    async def rerank(self, question: str, chunks: Sequence[RetrievedChunk]) -> list[int]:
        return list(range(len(chunks)))
