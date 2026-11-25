from collections.abc import Sequence
from dataclasses import dataclass

from grounded_rag.errors import BudgetExceededError
from grounded_rag.ingestion.embedder import Embedder, EmbeddingBatch
from grounded_rag.llm.client import ChatClient, ChatRequest, ChatResult


@dataclass(slots=True)
class SpendGuard:
    """Counts money spent on live provider calls and stops a run that passes its limit.

    It sits below the cache, so replayed answers are free and never count against the limit.
    The limit is checked before each call; one call can overshoot it.
    """

    max_usd: float | None = None
    spent_usd: float = 0.0
    live_calls: int = 0

    def check(self) -> None:
        if self.max_usd is not None and self.spent_usd >= self.max_usd:
            raise BudgetExceededError(
                f"Evaluation budget of ${self.max_usd:.2f} reached after {self.live_calls} live "
                f"calls (${self.spent_usd:.4f} spent). Raise --max-cost or run again: calls "
                "already made are cached and free."
            )

    def record(self, usd: float) -> None:
        self.live_calls += 1
        self.spent_usd += usd


class GuardedChat:
    def __init__(self, inner: ChatClient, guard: SpendGuard) -> None:
        self._inner = inner
        self._guard = guard

    async def complete(self, request: ChatRequest) -> ChatResult:
        self._guard.check()
        result = await self._inner.complete(request)
        self._guard.record(result.usage.cost_usd)
        return result


class GuardedEmbedder:
    def __init__(self, inner: Embedder, guard: SpendGuard) -> None:
        self._inner = inner
        self._guard = guard

    async def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self._guard.check()
        batch = await self._inner.embed(texts)
        self._guard.record(batch.usage.cost_usd)
        return batch
