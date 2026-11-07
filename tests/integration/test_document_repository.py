import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grounded_rag.db.repositories import DocumentRepository
from grounded_rag.domain.documents import (
    Chunk,
    DocumentStatus,
    NewDocument,
    content_sha256,
    make_chunk_id,
    make_document_id,
)

pytestmark = pytest.mark.db


def new_document(text: str = "alpha beta gamma", roles: frozenset[str] = frozenset({"legal"})):
    digest = content_sha256(text)
    return NewDocument(
        id=make_document_id("default", digest),
        corpus="default",
        source="docs/a.txt",
        title="A",
        content_hash=digest,
        allowed_roles=roles,
        text=text,
    )


def chunks_for(document: NewDocument, count: int = 2) -> list[Chunk]:
    return [
        Chunk(
            id=make_chunk_id(document.id, n),
            doc_id=document.id,
            ordinal=n,
            text=f"chunk {n}",
            start_char=n * 10,
            end_char=n * 10 + 7,
            section=None,
            context_header="A",
        )
        for n in range(count)
    ]


@pytest.fixture
def repository(session_factory: async_sessionmaker[AsyncSession]) -> DocumentRepository:
    return DocumentRepository(session_factory)


async def test_saved_document_starts_pending_and_keeps_its_chunks(
    repository: DocumentRepository,
) -> None:
    document = new_document()
    await repository.save_pending(document, chunks_for(document))

    stored = await repository.get(document.id)
    assert stored is not None
    assert stored.status is DocumentStatus.PENDING
    assert stored.chunk_count == 2
    assert await repository.chunk_ids(document.id) == [
        f"{document.id}-00000",
        f"{document.id}-00001",
    ]


async def test_saving_again_replaces_chunks_instead_of_duplicating_them(
    repository: DocumentRepository,
) -> None:
    document = new_document()
    await repository.save_pending(document, chunks_for(document, count=3))
    await repository.save_pending(document, chunks_for(document, count=2))

    assert len(await repository.chunks(document.id)) == 2


async def test_mark_ready_only_applies_to_pending_documents(
    repository: DocumentRepository,
) -> None:
    document = new_document()
    await repository.save_pending(document, chunks_for(document))

    assert await repository.mark_ready(document.id) is True
    assert await repository.mark_ready(document.id) is False
    stored = await repository.get(document.id)
    assert stored is not None
    assert stored.status is DocumentStatus.READY
    assert stored.ready_at is not None


async def test_document_being_deleted_cannot_become_ready(
    repository: DocumentRepository,
) -> None:
    document = new_document()
    await repository.save_pending(document, chunks_for(document))
    await repository.mark_deleting(document.id)

    assert await repository.mark_ready(document.id) is False


async def test_listing_returns_only_ready_documents_sharing_a_role(
    repository: DocumentRepository,
) -> None:
    visible = new_document("visible text", frozenset({"legal", "admin"}))
    other_role = new_document("other role text", frozenset({"finance"}))
    unfinished = new_document("unfinished text", frozenset({"legal"}))
    for document in (visible, other_role, unfinished):
        await repository.save_pending(document, chunks_for(document))
    await repository.mark_ready(visible.id)
    await repository.mark_ready(other_role.id)

    listed = await repository.list_visible("default", frozenset({"legal"}), limit=10)

    assert [doc.id for doc in listed] == [visible.id]


async def test_principal_without_roles_lists_nothing(repository: DocumentRepository) -> None:
    document = new_document()
    await repository.save_pending(document, chunks_for(document))
    await repository.mark_ready(document.id)

    assert await repository.list_visible("default", frozenset(), limit=10) == []


async def test_deleting_a_document_removes_its_chunks(repository: DocumentRepository) -> None:
    document = new_document()
    await repository.save_pending(document, chunks_for(document))
    await repository.delete(document.id)

    assert await repository.get(document.id) is None
    assert await repository.chunk_ids(document.id) == []


async def test_deleting_a_missing_document_is_not_an_error(repository: DocumentRepository) -> None:
    await repository.delete("does-not-exist")


async def test_updated_roles_change_who_can_list_the_document(
    repository: DocumentRepository,
) -> None:
    document = new_document(roles=frozenset({"legal"}))
    await repository.save_pending(document, chunks_for(document))
    await repository.mark_ready(document.id)
    await repository.update_roles(document.id, frozenset({"finance"}))

    assert await repository.list_visible("default", frozenset({"legal"}), limit=10) == []
    assert len(await repository.list_visible("default", frozenset({"finance"}), limit=10)) == 1


async def test_documents_of_another_corpus_are_not_listed(repository: DocumentRepository) -> None:
    document = new_document()
    await repository.save_pending(document, chunks_for(document))
    await repository.mark_ready(document.id)

    assert await repository.list_visible("other", frozenset({"legal"}), limit=10) == []


async def test_saving_never_reopens_a_document_that_is_already_ready(
    repository: DocumentRepository,
) -> None:
    document = new_document()
    await repository.save_pending(document, chunks_for(document, count=2))
    await repository.mark_ready(document.id)

    written = await repository.save_pending(document, chunks_for(document, count=3))

    stored = await repository.get(document.id)
    assert written is False
    assert stored is not None
    assert stored.status is DocumentStatus.READY
    assert len(await repository.chunks(document.id)) == 2


async def test_saving_never_resurrects_a_document_that_is_being_deleted(
    repository: DocumentRepository,
) -> None:
    document = new_document()
    await repository.save_pending(document, chunks_for(document))
    await repository.mark_deleting(document.id)

    assert await repository.save_pending(document, chunks_for(document)) is False

    stored = await repository.get(document.id)
    assert stored is not None
    assert stored.status is DocumentStatus.DELETING
