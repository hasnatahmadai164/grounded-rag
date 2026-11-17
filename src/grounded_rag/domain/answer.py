from dataclasses import dataclass
from enum import StrEnum


class AbstainReason(StrEnum):
    """Why no answer was given. Abstaining is a normal outcome, not an error."""

    NO_RELEVANT_CONTEXT = "no_relevant_context"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED_CITATIONS = "unsupported_citations"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass(frozen=True, slots=True)
class Citation:
    """A verified quote. Offsets locate it in the stored document text."""

    index: int
    chunk_id: str
    doc_id: str
    document_title: str
    section: str | None
    quote: str
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    abstained: bool
    abstain_reason: AbstainReason | None
    citations: tuple[Citation, ...] = ()

    @classmethod
    def abstention(cls, reason: AbstainReason) -> "Answer":
        return cls(text=ABSTENTION_TEXT, abstained=True, abstain_reason=reason)


ABSTENTION_TEXT = "The available documents do not contain enough evidence to answer this question."
