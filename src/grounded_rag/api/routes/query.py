import asyncio

from fastapi import APIRouter, Request

from grounded_rag.agent.service import AnswerResult
from grounded_rag.api.auth import PrincipalDep
from grounded_rag.api.deps import ContainerDep
from grounded_rag.api.errors import request_id_of
from grounded_rag.api.schemas import (
    CandidateOut,
    CitationOut,
    DebugOut,
    EventOut,
    QueryRequest,
    QueryResponse,
    StageOut,
    UsageOut,
)
from grounded_rag.errors import InvalidRequestError, RequestTimeoutError
from grounded_rag.logging import content_fields, get_logger

router = APIRouter(prefix="/v1", tags=["query"])
log = get_logger(__name__)


def to_debug(result: AnswerResult) -> DebugOut:
    """The retrieval trace. Every candidate in it has already passed the access check."""
    retrievals = []
    for retrieval in result.state.retrievals:
        stages = []
        for stage in retrieval.trace.stages:
            candidates = [
                CandidateOut(
                    chunk_id=hit.chunk_id,
                    document_title=retrieval.candidates[hit.chunk_id].document_title,
                    section=retrieval.candidates[hit.chunk_id].chunk.section,
                    score=hit.score,
                )
                for hit in stage.hits
                if hit.chunk_id in retrieval.candidates
            ]
            stages.append(
                StageOut(stage=stage.name, duration_ms=stage.duration_ms, candidates=candidates)
            )
        retrievals.append(stages)
    return DebugOut(
        queries=result.state.queries,
        retrievals=retrievals,
        events=[EventOut(node=e.node, detail=e.detail) for e in result.state.events],
        model_calls=result.llm_calls,
        rewrites_used=result.state.rewrites_used,
        repairs_used=result.state.repairs_used,
    )


@router.post("/query")
async def query(
    body: QueryRequest, request: Request, principal: PrincipalDep, container: ContainerDep
) -> QueryResponse:
    settings = container.settings
    if body.top_k is not None and body.top_k > settings.max_top_k:
        raise InvalidRequestError(
            f"top_k must be at most {settings.max_top_k}.",
            details={"max_top_k": settings.max_top_k},
        )
    log.info(
        "query_received",
        **content_fields("question", body.question, include_content=settings.log_content),
    )
    try:
        async with asyncio.timeout(settings.request_timeout_seconds):
            result = await container.answers.answer(
                body.question, principal, corpus=settings.default_corpus, top_k=body.top_k
            )
    except TimeoutError as error:
        raise RequestTimeoutError(
            f"The request took longer than {settings.request_timeout_seconds:g} seconds."
        ) from error

    answer, usage = result.answer, result.usage
    return QueryResponse(
        answer=answer.text,
        abstained=answer.abstained,
        abstain_reason=answer.abstain_reason.value if answer.abstain_reason else None,
        citations=[CitationOut(**vars_of(c)) for c in answer.citations],
        usage=UsageOut(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            embedding_tokens=usage.embedding_tokens,
            cost_usd=usage.cost_usd,
            latency_ms=round(result.latency_ms, 1),
        ),
        request_id=request_id_of(request),
        debug=to_debug(result) if body.debug else None,
    )


def vars_of(citation: object) -> dict[str, object]:
    return {name: getattr(citation, name) for name in CitationOut.model_fields}
