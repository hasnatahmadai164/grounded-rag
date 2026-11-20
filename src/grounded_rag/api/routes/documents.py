from typing import Annotated

from fastapi import APIRouter, Query

from grounded_rag.api.auth import PrincipalDep
from grounded_rag.api.deps import ContainerDep
from grounded_rag.api.schemas import DocumentList, DocumentOut

router = APIRouter(prefix="/v1", tags=["documents"])


@router.get("/documents")
async def list_documents(
    principal: PrincipalDep,
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    """Documents in the default corpus that the caller's roles may read."""
    documents = await container.documents.list_visible(
        container.settings.default_corpus, principal.roles, limit=limit, offset=offset
    )
    return DocumentList(
        documents=[
            DocumentOut(
                id=d.id,
                title=d.title,
                chunk_count=d.chunk_count,
                char_count=d.char_count,
                created_at=d.created_at,
            )
            for d in documents
        ],
        limit=limit,
        offset=offset,
    )
