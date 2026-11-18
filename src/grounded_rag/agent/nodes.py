from collections.abc import Sequence

from grounded_rag.agent.assemble import build_answer
from grounded_rag.agent.config import AgentConfig
from grounded_rag.agent.schemas import GeneratedAnswer, GradeVerdict
from grounded_rag.agent.state import AgentState, TraceEvent
from grounded_rag.agent.verify import VerificationReport, verify_claims
from grounded_rag.domain.answer import AbstainReason, Answer
from grounded_rag.domain.documents import RetrievedChunk
from grounded_rag.errors import BudgetExceededError
from grounded_rag.llm.budget import MeteredLLM, UsageMeter
from grounded_rag.llm.prompts import (
    generate_messages,
    grade_messages,
    passage_label,
    repair_messages,
)
from grounded_rag.retrieval.rerank import Reranker
from grounded_rag.retrieval.rewrite import QueryRewriter
from grounded_rag.retrieval.service import RetrievalRequest, RetrievalService

Update = dict[str, object]


def same_query(left: str, right: str) -> bool:
    return " ".join(left.split()).casefold() == " ".join(right.split()).casefold()


def labelled(context: Sequence[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    return {passage_label(n): item for n, item in enumerate(context, start=1)}


def describe_failures(report: VerificationReport) -> list[str]:
    """One line per failed citation, in terms the generator can act on."""
    problems = []
    for number, claim in enumerate(report.claims, start=1):
        for check in claim.checks:
            if not check.ok:
                quote = check.citation.quote[:120]
                problems.append(
                    f'claim {number}: {check.problem} for {check.citation.passage_id}: "{quote}"'
                )
        if not claim.checks:
            problems.append(f"claim {number}: no citations")
    return problems


def merge_context(
    fresh: Sequence[RetrievedChunk], earlier: Sequence[RetrievedChunk], limit: int
) -> list[RetrievedChunk]:
    """New results first, then anything found by earlier attempts, without repeats."""
    merged: dict[str, RetrievedChunk] = {}
    for item in [*fresh, *earlier]:
        merged.setdefault(item.chunk.id, item)
    return list(merged.values())[:limit]


class AgentNodes:
    """The steps of the answer loop. One instance serves one request."""

    def __init__(
        self,
        *,
        retrieval: RetrievalService,
        llm: MeteredLLM,
        meter: UsageMeter,
        reranker: Reranker | None,
        rewriter: QueryRewriter,
        config: AgentConfig,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._meter = meter
        self._reranker = reranker
        self._rewriter = rewriter
        self._config = config

    @staticmethod
    def _out_of_budget(state: AgentState, node: str, error: BudgetExceededError) -> Update:
        event = TraceEvent(node, {"stopped": error.message})
        return {"stop_reason": AbstainReason.BUDGET_EXCEEDED, "events": [*state.events, event]}

    async def retrieve(self, state: AgentState) -> Update:
        query = state.queries[-1]
        request = RetrievalRequest(
            question=query,
            principal=state.principal,
            corpus=state.corpus,
            mode=self._config.mode,
            top_k=state.top_k,
        )
        try:
            result = await self._retrieval.retrieve(request, reranker=self._reranker)
        except BudgetExceededError as error:
            return self._out_of_budget(state, "retrieve", error)
        self._meter.record_embedding(result.usage, result.replay_credit_ms)
        context = merge_context(result.chunks, state.context, self._config.max_context_chunks)
        event = TraceEvent("retrieve", {"query": query, "chunks": len(result.chunks)})
        return {
            "context": context,
            "retrievals": [*state.retrievals, result],
            "events": [*state.events, event],
        }

    async def grade(self, state: AgentState) -> Update:
        if not state.context:
            verdict = GradeVerdict(
                sufficient=False,
                reason="No readable passages were retrieved.",
                missing="everything",
            )
        else:
            try:
                verdict = await self._llm.structured(
                    grade_messages(state.question, state.context),
                    GradeVerdict,
                    purpose="grade",
                    max_output_tokens=200,
                )
            except BudgetExceededError as error:
                return self._out_of_budget(state, "grade", error)
        event = TraceEvent("grade", {"sufficient": verdict.sufficient, "reason": verdict.reason})
        return {"grade": verdict, "events": [*state.events, event]}

    async def rewrite(self, state: AgentState) -> Update:
        assert state.grade is not None
        feedback = f"{state.grade.reason} Missing: {state.grade.missing or 'unspecified'}"
        try:
            query = await self._rewriter.rewrite(state.question, state.queries, feedback)
        except BudgetExceededError as error:
            return self._out_of_budget(state, "rewrite", error)
        if any(same_query(query, earlier) for earlier in state.queries):
            event = TraceEvent("rewrite", {"query": query, "repeated": True})
            return {"stalled": True, "events": [*state.events, event]}
        event = TraceEvent("rewrite", {"query": query})
        return {
            "queries": [*state.queries, query],
            "rewrites_used": state.rewrites_used + 1,
            "events": [*state.events, event],
        }

    async def generate(self, state: AgentState) -> Update:
        try:
            generated = await self._llm.structured(
                generate_messages(state.question, state.context),
                GeneratedAnswer,
                purpose="generate",
            )
        except BudgetExceededError as error:
            return self._out_of_budget(state, "generate", error)
        detail = {"answerable": generated.answerable, "claims": len(generated.claims)}
        return {"generated": generated, "events": [*state.events, TraceEvent("generate", detail)]}

    async def verify(self, state: AgentState) -> Update:
        assert state.generated is not None
        report = verify_claims(state.generated.claims, labelled(state.context))
        detail = {"citations": len(report.checks), "failed": len(report.failures)}
        return {
            "reports": [*state.reports, report],
            "events": [*state.events, TraceEvent("verify", detail)],
        }

    async def repair(self, state: AgentState) -> Update:
        assert state.generated is not None
        problems = describe_failures(state.reports[-1])
        attempt = state.repairs_used + 1
        try:
            repaired = await self._llm.structured(
                repair_messages(
                    state.question,
                    state.context,
                    state.generated.model_dump_json(),
                    problems,
                ),
                GeneratedAnswer,
                purpose="repair",
            )
        except BudgetExceededError as error:
            # Keep what we have: unsupported claims are dropped at the end.
            event = TraceEvent("repair", {"stopped": error.message})
            return {"repairs_used": attempt, "events": [*state.events, event]}
        event = TraceEvent("repair", {"problems": len(problems)})
        return {"generated": repaired, "repairs_used": attempt, "events": [*state.events, event]}

    async def finalize(self, state: AgentState) -> Update:
        assert state.reports
        survivors = state.reports[-1].supported_claims()
        if not survivors:
            answer = Answer.abstention(AbstainReason.UNSUPPORTED_CITATIONS)
        else:
            answer = build_answer(survivors, labelled(state.context))
        detail = {"abstained": answer.abstained, "citations": len(answer.citations)}
        return {"answer": answer, "events": [*state.events, TraceEvent("finalize", detail)]}

    async def abstain(self, state: AgentState) -> Update:
        reason = state.stop_reason or (
            AbstainReason.INSUFFICIENT_EVIDENCE
            if state.context
            else AbstainReason.NO_RELEVANT_CONTEXT
        )
        event = TraceEvent("abstain", {"reason": reason.value})
        return {"answer": Answer.abstention(reason), "events": [*state.events, event]}
