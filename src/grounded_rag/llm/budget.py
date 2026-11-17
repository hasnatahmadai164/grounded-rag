from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from grounded_rag.domain.usage import Usage
from grounded_rag.errors import BudgetExceededError
from grounded_rag.llm.client import (
    ChatClient,
    ChatMessage,
    ChatRequest,
    ChatResult,
    parse_structured,
    structured_request,
)


@dataclass(frozen=True, slots=True)
class Budget:
    max_tokens: int
    max_cost_usd: float


@dataclass(slots=True)
class UsageMeter:
    """Spend and time for one request, broken down by what the calls were for."""

    usage: Usage = field(default_factory=Usage)
    by_purpose: dict[str, Usage] = field(default_factory=dict)
    calls: int = 0
    model_latency_ms: float = 0.0
    replay_credit_ms: float = 0.0

    def record_chat(self, purpose: str, result: ChatResult) -> None:
        self.calls += 1
        self.model_latency_ms += result.latency_ms
        self.replay_credit_ms += result.replay_credit_ms
        self._add(purpose, result.usage)

    def record_embedding(self, usage: Usage, replay_credit_ms: float = 0.0) -> None:
        self.replay_credit_ms += replay_credit_ms
        self._add("embedding", usage)

    def _add(self, purpose: str, usage: Usage) -> None:
        self.usage = self.usage + usage
        self.by_purpose[purpose] = self.by_purpose.get(purpose, Usage()) + usage


class MeteredLLM:
    """The chat client as one request sees it: every call is checked against the budget
    first and recorded afterwards."""

    def __init__(self, client: ChatClient, meter: UsageMeter, budget: Budget) -> None:
        self._client = client
        self._meter = meter
        self._budget = budget

    def ensure_available(self) -> None:
        usage, budget = self._meter.usage, self._budget
        if usage.total_tokens >= budget.max_tokens:
            raise BudgetExceededError(
                f"Token budget reached: {usage.total_tokens} of {budget.max_tokens} used."
            )
        if usage.cost_usd >= budget.max_cost_usd:
            raise BudgetExceededError(
                f"Cost budget reached: ${usage.cost_usd:.4f} of ${budget.max_cost_usd:.4f} spent."
            )

    async def complete(self, request: ChatRequest) -> ChatResult:
        self.ensure_available()
        result = await self._client.complete(request)
        self._meter.record_chat(request.purpose, result)
        return result

    async def structured[T: BaseModel](
        self,
        messages: Sequence[ChatMessage],
        output: type[T],
        *,
        purpose: str,
        max_output_tokens: int | None = None,
    ) -> T:
        request = structured_request(
            messages, output, purpose=purpose, max_output_tokens=max_output_tokens
        )
        result = await self.complete(request)
        return parse_structured(result, output)
