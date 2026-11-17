import pytest
from pydantic import BaseModel

from grounded_rag.domain.usage import Usage
from grounded_rag.errors import BudgetExceededError, LLMOutputError
from grounded_rag.llm.budget import Budget, MeteredLLM, UsageMeter
from grounded_rag.llm.client import ChatMessage, ChatRequest, ChatResult


class Answer(BaseModel):
    text: str


class ScriptedClient:
    def __init__(
        self, *contents: str, prompt: int = 100, completion: int = 50, cost: float = 0.001
    ):
        self._contents = list(contents)
        self._usage = Usage(prompt_tokens=prompt, completion_tokens=completion, cost_usd=cost)
        self.requests: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        return ChatResult(self._contents.pop(0), self._usage, latency_ms=12.0, model="m")


def llm_for(client: ScriptedClient, meter: UsageMeter, **budget: float) -> MeteredLLM:
    limits = {"max_tokens": 1000, "max_cost_usd": 1.0} | budget
    return MeteredLLM(client, meter, Budget(int(limits["max_tokens"]), limits["max_cost_usd"]))


MESSAGES = [ChatMessage("user", "hi")]


async def test_each_call_is_recorded_under_its_purpose() -> None:
    meter = UsageMeter()
    llm = llm_for(ScriptedClient('{"text": "a"}', '{"text": "b"}', '{"text": "c"}'), meter)

    await llm.structured(MESSAGES, Answer, purpose="grade")
    await llm.structured(MESSAGES, Answer, purpose="grade")
    await llm.structured(MESSAGES, Answer, purpose="generate")

    assert meter.calls == 3
    assert meter.by_purpose["grade"].prompt_tokens == 200
    assert meter.by_purpose["generate"].prompt_tokens == 100
    assert meter.usage.total_tokens == 450
    assert meter.model_latency_ms == 36.0


async def test_a_call_over_the_token_budget_is_refused_before_it_is_made() -> None:
    meter = UsageMeter()
    client = ScriptedClient("x", "x", prompt=300, completion=100)
    llm = llm_for(client, meter, max_tokens=500)

    await llm.complete(ChatRequest(MESSAGES))
    await llm.complete(ChatRequest(MESSAGES))

    with pytest.raises(BudgetExceededError, match="Token budget"):
        await llm.complete(ChatRequest(MESSAGES))
    assert len(client.requests) == 2


async def test_a_call_over_the_cost_budget_is_refused() -> None:
    meter = UsageMeter()
    llm = llm_for(ScriptedClient("x", "x", cost=0.03), meter, max_cost_usd=0.05)

    await llm.complete(ChatRequest(MESSAGES))
    await llm.complete(ChatRequest(MESSAGES))

    with pytest.raises(BudgetExceededError, match="Cost budget"):
        await llm.complete(ChatRequest(MESSAGES))


async def test_embedding_spend_counts_against_the_same_budget() -> None:
    meter = UsageMeter()
    meter.record_embedding(Usage(embedding_tokens=900, cost_usd=0.0))
    llm = llm_for(ScriptedClient("x"), meter, max_tokens=800)

    with pytest.raises(BudgetExceededError):
        await llm.complete(ChatRequest(MESSAGES))


async def test_a_call_that_returns_invalid_output_still_counts_toward_spend() -> None:
    meter = UsageMeter()
    llm = llm_for(ScriptedClient("not json"), meter)

    with pytest.raises(LLMOutputError):
        await llm.structured(MESSAGES, Answer, purpose="generate")

    assert meter.calls == 1
    assert meter.usage.total_tokens == 150


async def test_cache_replay_credit_is_accumulated() -> None:
    meter = UsageMeter()
    meter.record_chat(
        "grade",
        ChatResult("x", Usage(), latency_ms=1.0, model="m", cached=True, replay_credit_ms=800.0),
    )
    assert meter.replay_credit_ms == 800.0
