from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentOut(BaseModel):
    id: str
    title: str
    chunk_count: int
    char_count: int
    created_at: datetime


class DocumentList(BaseModel):
    documents: list[DocumentOut]
    limit: int
    offset: int


class QueryRequest(BaseModel):
    question: str = Field(max_length=2000)
    top_k: int | None = Field(default=None, ge=1)
    debug: bool = False

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    document_title: str
    section: str | None
    quote: str
    start_char: int
    end_char: int


class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    embedding_tokens: int
    cost_usd: float
    latency_ms: float


class CandidateOut(BaseModel):
    chunk_id: str
    document_title: str
    section: str | None
    score: float


class StageOut(BaseModel):
    stage: str
    duration_ms: float
    candidates: list[CandidateOut]


class EventOut(BaseModel):
    node: str
    detail: dict[str, Any]


class DebugOut(BaseModel):
    queries: list[str]
    retrievals: list[list[StageOut]]
    events: list[EventOut]
    model_calls: int
    rewrites_used: int
    repairs_used: int


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    abstain_reason: str | None
    citations: list[CitationOut]
    usage: UsageOut
    request_id: str
    debug: DebugOut | None = None
