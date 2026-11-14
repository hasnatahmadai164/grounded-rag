import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grounded_rag.db.repositories import ChunkRepository, DocumentRepository
from grounded_rag.domain.documents import make_chunk_id
from tests.integration.seed import seed_document

pytestmark = pytest.mark.db


@pytest.fixture
def documents(session_factory: async_sessionmaker[AsyncSession]) -> DocumentRepository:
    return DocumentRepository(session_factory)


@pytest.fixture
def chunks(session_factory: async_sessionmaker[AsyncSession]) -> ChunkRepository:
    return ChunkRepository(session_factory)


async def test_readable_chunks_are_returned_with_their_document_facts(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    doc = await seed_document(
        documents, title="MSA", chunk_texts=["one", "two"], roles=frozenset({"legal", "admin"})
    )

    found = await chunks.visible_chunks(
        "default", frozenset({"legal"}), [make_chunk_id(doc.id, 0), make_chunk_id(doc.id, 1)]
    )

    assert {item.chunk.text for item in found} == {"one", "two"}
    assert all(item.document_title == "MSA" for item in found)
    assert all(item.allowed_roles == frozenset({"legal", "admin"}) for item in found)


async def test_chunks_of_unreadable_documents_are_not_returned(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    doc = await seed_document(
        documents, title="Secret", chunk_texts=["x"], roles=frozenset({"finance"})
    )

    assert (
        await chunks.visible_chunks("default", frozenset({"legal"}), [make_chunk_id(doc.id, 0)])
        == []
    )


async def test_roles_changed_after_indexing_take_effect_immediately(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    doc = await seed_document(documents, title="Doc", chunk_texts=["x"], roles=frozenset({"legal"}))
    chunk_id = make_chunk_id(doc.id, 0)
    assert len(await chunks.visible_chunks("default", frozenset({"legal"}), [chunk_id])) == 1

    await documents.update_roles(doc.id, frozenset({"finance"}))

    assert await chunks.visible_chunks("default", frozenset({"legal"}), [chunk_id]) == []


@pytest.mark.parametrize("status", ["pending", "deleting"])
async def test_documents_that_are_not_ready_are_not_visible(
    documents: DocumentRepository, chunks: ChunkRepository, status: str
) -> None:
    doc = await seed_document(documents, title="Doc", chunk_texts=["x"], ready=status != "pending")
    if status == "deleting":
        await documents.mark_deleting(doc.id)

    assert (
        await chunks.visible_chunks("default", frozenset({"legal"}), [make_chunk_id(doc.id, 0)])
        == []
    )


async def test_other_corpora_and_empty_inputs_return_nothing(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    doc = await seed_document(documents, title="Doc", chunk_texts=["x"], corpus="other")
    chunk_id = make_chunk_id(doc.id, 0)

    assert await chunks.visible_chunks("default", frozenset({"legal"}), [chunk_id]) == []
    assert await chunks.visible_chunks("other", frozenset(), [chunk_id]) == []
    assert await chunks.visible_chunks("other", frozenset({"legal"}), []) == []
