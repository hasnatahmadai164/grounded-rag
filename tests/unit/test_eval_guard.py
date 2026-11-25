import pytest

from grounded_rag.domain.usage import Usage
from grounded_rag.errors import BudgetExceededError
from grounded_rag.evals.guard import GuardedChat, GuardedEmbedder, SpendGuard
from grounded_rag.llm.client import ChatRequest, ChatResult
from tests.fakes import FakeEmbedder


class PricedChat:
    def __init__(self, cost: float) -> None:
        self.cost = cost
        self.calls = 0

    async def complete(self, request: ChatRequest) -> ChatResult:
        self.calls += 1
        return ChatResult("x", Usage(cost_usd=self.cost), 1.0, "m")


async def test_live_calls_are_counted_and_priced() -> None:
    guard = SpendGuard(max_usd=1.0)
    chat = GuardedChat(PricedChat(0.25), guard)

    await chat.complete(ChatRequest(()))
    await chat.complete(ChatRequest(()))

    assert (guard.live_calls, guard.spent_usd) == (2, 0.5)


async def test_a_run_over_its_limit_stops_before_making_another_call() -> None:
    guard = SpendGuard(max_usd=0.5)
    inner = PricedChat(0.3)
    chat = GuardedChat(inner, guard)

    await chat.complete(ChatRequest(()))
    await chat.complete(ChatRequest(()))

    with pytest.raises(BudgetExceededError, match="Raise --max-cost"):
        await chat.complete(ChatRequest(()))
    assert inner.calls == 2


async def test_no_limit_means_no_stop() -> None:
    guard = SpendGuard(max_usd=None)
    chat = GuardedChat(PricedChat(100.0), guard)
    for _ in range(3):
        await chat.complete(ChatRequest(()))
    assert guard.live_calls == 3


async def test_embeddings_count_against_the_same_limit() -> None:
    guard = SpendGuard(max_usd=0.0)
    embedder = GuardedEmbedder(FakeEmbedder(), guard)

    with pytest.raises(BudgetExceededError):
        await embedder.embed(["text"])
