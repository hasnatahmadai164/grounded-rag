import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grounded_rag.db.repositories import ChunkRepository, DocumentRepository
from grounded_rag.domain.documents import make_chunk_id
from grounded_rag.domain.principal import Principal
from grounded_rag.retrieval.dense import DenseRetriever
from grounded_rag.retrieval.service import RetrievalRequest, RetrievalService
from grounded_rag.vectorstore.base import VectorRecord
from tests.fakes import FakeEmbedder, InMemoryVectorStore, hashed_vector
from tests.integration.seed import seed_document

pytestmark = pytest.mark.db

LEGAL = Principal("legal-analyst", frozenset({"legal"}))
SECRET = "The confidential revenue share is twelve percent."


async def test_postgres_blocks_a_chunk_that_the_vector_store_still_advertises(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    documents = DocumentRepository(session_factory)
    document = await seed_document(
        documents, title="Finance Memo", chunk_texts=[SECRET], roles=frozenset({"finance"})
    )
    vectors = InMemoryVectorStore()
    await vectors.upsert(
        "default",
        [
            VectorRecord(
                id=make_chunk_id(document.id, 0),
                values=hashed_vector(f"Finance Memo\n\n{SECRET}"),
                doc_id=document.id,
                allowed_roles=frozenset({"legal"}),
            )
        ],
    )
    service = RetrievalService(
        DenseRetriever(FakeEmbedder(), vectors), ChunkRepository(session_factory)
    )

    result = await service.retrieve(RetrievalRequest("revenue share", LEGAL, "default"))

    assert result.chunks == []
    assert result.dropped_unauthorized == 1
    assert all(stage.hits == () for stage in result.trace.stages)


async def test_end_to_end_retrieval_returns_only_readable_chunks_from_postgres(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    documents = DocumentRepository(session_factory)
    readable = await seed_document(
        documents,
        title="Legal Memo",
        chunk_texts=["The governing law is Delaware."],
        roles=frozenset({"legal"}),
    )
    await seed_document(
        documents, title="Finance Memo", chunk_texts=[SECRET], roles=frozenset({"finance"})
    )
    service = RetrievalService(
        DenseRetriever(FakeEmbedder(), InMemoryVectorStore()), ChunkRepository(session_factory)
    )

    result = await service.retrieve(
        RetrievalRequest("revenue share governing law", LEGAL, "default")
    )

    assert [item.chunk.id for item in result.chunks] == [make_chunk_id(readable.id, 0)]
