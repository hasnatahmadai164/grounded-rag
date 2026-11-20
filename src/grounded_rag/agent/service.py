import time
from dataclasses import dataclass

from grounded_rag.agent.config import AgentConfig
from grounded_rag.agent.graph import build_graph, run_graph
from grounded_rag.agent.nodes import AgentNodes
from grounded_rag.agent.state import AgentState
from grounded_rag.domain.answer import Answer
from grounded_rag.domain.principal import Principal
from grounded_rag.domain.usage import Usage
from grounded_rag.llm.budget import Budget, MeteredLLM, UsageMeter
from grounded_rag.llm.client import ChatClient
from grounded_rag.logging import get_logger
from grounded_rag.retrieval.rerank import LLMReranker
from grounded_rag.retrieval.rewrite import QueryRewriter
from grounded_rag.retrieval.service import RetrievalService

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """The answer plus everything measured while producing it."""

    answer: Answer
    usage: Usage
    by_purpose: dict[str, Usage]
    llm_calls: int
    latency_ms: float
    replay_credit_ms: float
    state: AgentState
    config: AgentConfig

    @property
    def reported_latency_ms(self) -> float:
        """Wall time, plus what calls replayed from a cache originally took."""
        return self.latency_ms + self.replay_credit_ms


class AnswerService:
    """Answers one question for one principal. Safe to share: all request state is local."""

    def __init__(
        self,
        retrieval: RetrievalService,
        chat: ChatClient,
        budget: Budget,
        defaults: AgentConfig | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._chat = chat
        self._budget = budget
        self._defaults = defaults or AgentConfig()

    async def answer(
        self,
        question: str,
        principal: Principal,
        *,
        corpus: str,
        top_k: int | None = None,
        config: AgentConfig | None = None,
    ) -> AnswerResult:
        config = config or self._defaults
        meter = UsageMeter()
        llm = MeteredLLM(self._chat, meter, self._budget)
        nodes = AgentNodes(
            retrieval=self._retrieval,
            llm=llm,
            meter=meter,
            reranker=LLMReranker(llm) if config.rerank else None,
            rewriter=QueryRewriter(llm),
            config=config,
        )
        state = AgentState(
            question=question,
            principal=principal,
            corpus=corpus,
            top_k=top_k or config.top_k,
            queries=[question],
        )
        started = time.perf_counter()
        final = await run_graph(build_graph(nodes, config), state)
        latency_ms = (time.perf_counter() - started) * 1000
        assert final.answer is not None, "every path through the graph ends with an answer"
        log.info(
            "question_answered",
            principal=principal.name,
            abstained=final.answer.abstained,
            abstain_reason=final.answer.abstain_reason,
            citations=len(final.answer.citations),
            rewrites=final.rewrites_used,
            repairs=final.repairs_used,
            llm_calls=meter.calls,
            tokens=meter.usage.total_tokens,
            cost_usd=meter.usage.cost_usd,
            latency_ms=round(latency_ms, 1),
        )
        return AnswerResult(
            answer=final.answer,
            usage=meter.usage,
            by_purpose=dict(meter.by_purpose),
            llm_calls=meter.calls,
            latency_ms=latency_ms,
            replay_credit_ms=meter.replay_credit_ms,
            state=final,
            config=config,
        )
