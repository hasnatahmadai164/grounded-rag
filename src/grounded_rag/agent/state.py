from dataclasses import dataclass, field
from typing import Any

from grounded_rag.agent.schemas import GeneratedAnswer, GradeVerdict
from grounded_rag.agent.verify import VerificationReport
from grounded_rag.domain.answer import AbstainReason, Answer
from grounded_rag.domain.documents import RetrievedChunk
from grounded_rag.domain.principal import Principal
from grounded_rag.domain.retrieval import RetrievalResult


@dataclass(frozen=True, slots=True)
class TraceEvent:
    node: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Everything the loop knows. Nodes return updates to these fields."""

    question: str
    principal: Principal
    corpus: str
    top_k: int
    queries: list[str]
    context: list[RetrievedChunk] = field(default_factory=list)
    retrievals: list[RetrievalResult] = field(default_factory=list)
    rewrites_used: int = 0
    stalled: bool = False
    grade: GradeVerdict | None = None
    generated: GeneratedAnswer | None = None
    repairs_used: int = 0
    reports: list[VerificationReport] = field(default_factory=list)
    stop_reason: AbstainReason | None = None
    answer: Answer | None = None
    events: list[TraceEvent] = field(default_factory=list)
