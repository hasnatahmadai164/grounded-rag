from dataclasses import dataclass

from grounded_rag.domain.retrieval import RetrievalMode


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Behaviour of the answer loop. Evaluation ablations are different values of this."""

    mode: RetrievalMode = RetrievalMode.HYBRID
    rerank: bool = True
    max_rewrites: int = 2
    max_repairs: int = 1
    top_k: int = 6
    max_context_chunks: int = 10
