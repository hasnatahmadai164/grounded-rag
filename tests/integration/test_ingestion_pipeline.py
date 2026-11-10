import asyncio
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grounded_rag.db.repositories import DocumentRepository
from grounded_rag.domain.documents import DocumentStatus
from grounded_rag.errors import EmptyDocumentError, VectorStoreError
from grounded_rag.ingestion.chunking import ChunkingConfig
from grounded_rag.ingestion.pipeline import IngestionPipeline, IngestOutcome
from tests.fakes import FakeEmbedder, InMemoryVectorStore

pytestmark = pytest.mark.db

SMALL = ChunkingConfig(target_chars=300, max_chars=500, min_chars=50, overlap_chars=40)
CONTRACT = (
    "1. Term\n\n" + "The agreement continues for five years. " * 12 + "\n\n"
    "2. Payment\n\n" + "Fees are due within thirty days of invoice. " * 12
)


class Harness:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.documents = DocumentRepository(sessions)
        self.embedder = FakeEmbedder()
        self.vectors = InMemoryVectorStore()
        self.pipeline = IngestionPipeline(self.documents, self.embedder, self.vectors, SMALL)


@pytest.fixture
def harness(session_factory: async_sessionmaker[AsyncSession]) -> Harness:
    return Harness(session_factory)


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


async def test_ingested_document_is_ready_with_a_vector_per_chunk(
    harness: Harness, tmp_path: Path
) -> None:
    result = await harness.pipeline.ingest(
        write(tmp_path, "msa.txt", CONTRACT), roles=frozenset({"legal"}), corpus="c"
    )

    document = await harness.documents.get(result.doc_id)
    assert document is not None
    assert document.status is DocumentStatus.READY
    assert result.outcome is IngestOutcome.CREATED
    assert await harness.vectors.count("c") == result.chunk_count == document.chunk_count
    assert result.usage.embedding_tokens > 0


async def test_chunk_offsets_point_at_the_stored_document_text(
    harness: Harness, tmp_path: Path
) -> None:
    result = await harness.pipeline.ingest(
        write(tmp_path, "msa.txt", CONTRACT), roles=frozenset({"legal"}), corpus="c"
    )

    text = await harness.documents.get_text(result.doc_id)
    chunks = await harness.documents.chunks(result.doc_id)
    assert text is not None
    assert len(chunks) > 1
    assert all(text[c.start_char : c.end_char] == c.text for c in chunks)
    assert {c.section for c in chunks} == {"1. Term", "2. Payment"}


async def test_vectors_carry_the_documents_roles(harness: Harness, tmp_path: Path) -> None:
    result = await harness.pipeline.ingest(
        write(tmp_path, "msa.txt", CONTRACT), roles=frozenset({"legal", "admin"}), corpus="c"
    )

    chunk_ids = await harness.documents.chunk_ids(result.doc_id)
    assert harness.vectors.roles_of("c", chunk_ids[0]) == frozenset({"legal", "admin"})


async def test_embedding_input_includes_title_and_section(harness: Harness, tmp_path: Path) -> None:
    await harness.pipeline.ingest(
        write(tmp_path, "Supply Agreement.txt", CONTRACT), roles=frozenset({"legal"}), corpus="c"
    )

    first_input = harness.embedder.calls[0][0]
    assert first_input.startswith("Supply Agreement > 1. Term")


async def test_ingesting_the_same_content_again_does_nothing(
    harness: Harness, tmp_path: Path
) -> None:
    path = write(tmp_path, "msa.txt", CONTRACT)
    await harness.pipeline.ingest(path, roles=frozenset({"legal"}), corpus="c")
    calls_before = len(harness.embedder.calls)

    again = await harness.pipeline.ingest(path, roles=frozenset({"legal"}), corpus="c")

    assert again.outcome is IngestOutcome.UNCHANGED
    assert len(harness.embedder.calls) == calls_before
    assert again.usage.cost_usd == 0


async def test_same_content_in_another_corpus_is_a_separate_document(
    harness: Harness, tmp_path: Path
) -> None:
    path = write(tmp_path, "msa.txt", CONTRACT)
    first = await harness.pipeline.ingest(path, roles=frozenset({"legal"}), corpus="a")
    second = await harness.pipeline.ingest(path, roles=frozenset({"legal"}), corpus="b")

    assert first.doc_id != second.doc_id


async def test_changing_roles_updates_postgres_and_the_vectors(
    harness: Harness, tmp_path: Path
) -> None:
    path = write(tmp_path, "msa.txt", CONTRACT)
    first = await harness.pipeline.ingest(path, roles=frozenset({"legal"}), corpus="c")

    again = await harness.pipeline.ingest(path, roles=frozenset({"finance"}), corpus="c")

    document = await harness.documents.get(first.doc_id)
    chunk_ids = await harness.documents.chunk_ids(first.doc_id)
    assert again.outcome is IngestOutcome.ROLES_UPDATED
    assert document is not None
    assert document.allowed_roles == frozenset({"finance"})
    assert harness.vectors.roles_of("c", chunk_ids[0]) == frozenset({"finance"})
    assert await harness.vectors.count("c") == len(chunk_ids)


async def test_failed_upsert_leaves_a_pending_document_that_rerunning_completes(
    harness: Harness, tmp_path: Path
) -> None:
    path = write(tmp_path, "msa.txt", CONTRACT)
    harness.vectors.fail_next_upsert = VectorStoreError("Pinecone returned 503")

    with pytest.raises(VectorStoreError):
        await harness.pipeline.ingest(path, roles=frozenset({"legal"}), corpus="c")

    pending = await harness.documents.list_by_source("c", path.as_posix())
    assert [d.status for d in pending] == [DocumentStatus.PENDING]
    assert await harness.documents.list_visible("c", frozenset({"legal"}), limit=10) == []

    result = await harness.pipeline.ingest(path, roles=frozenset({"legal"}), corpus="c")

    assert result.outcome is IngestOutcome.RESUMED
    assert await harness.vectors.count("c") == result.chunk_count
    assert len(await harness.documents.chunks(result.doc_id)) == result.chunk_count


async def test_delete_removes_vectors_and_rows(harness: Harness, tmp_path: Path) -> None:
    result = await harness.pipeline.ingest(
        write(tmp_path, "msa.txt", CONTRACT), roles=frozenset({"legal"}), corpus="c"
    )

    assert await harness.pipeline.delete(result.doc_id) is True

    assert await harness.vectors.count("c") == 0
    assert await harness.documents.get(result.doc_id) is None


async def test_deleting_twice_is_harmless(harness: Harness, tmp_path: Path) -> None:
    result = await harness.pipeline.ingest(
        write(tmp_path, "msa.txt", CONTRACT), roles=frozenset({"legal"}), corpus="c"
    )
    await harness.pipeline.delete(result.doc_id)

    assert await harness.pipeline.delete(result.doc_id) is False


async def test_failed_vector_delete_keeps_the_document_hidden_until_retry(
    harness: Harness, tmp_path: Path
) -> None:
    result = await harness.pipeline.ingest(
        write(tmp_path, "msa.txt", CONTRACT), roles=frozenset({"legal"}), corpus="c"
    )
    harness.vectors.fail_next_delete = VectorStoreError("Pinecone returned 500")

    with pytest.raises(VectorStoreError):
        await harness.pipeline.delete(result.doc_id)

    document = await harness.documents.get(result.doc_id)
    assert document is not None
    assert document.status is DocumentStatus.DELETING
    assert await harness.documents.list_visible("c", frozenset({"legal"}), limit=10) == []

    assert await harness.pipeline.delete(result.doc_id) is True
    assert await harness.vectors.count("c") == 0


async def test_replace_removes_the_previous_version_of_the_same_source(
    harness: Harness, tmp_path: Path
) -> None:
    path = write(tmp_path, "msa.txt", CONTRACT)
    old = await harness.pipeline.ingest(path, roles=frozenset({"legal"}), corpus="c")
    path.write_text(CONTRACT + "\n\n3. Notices\n\nNotices must be written.", encoding="utf-8")

    new = await harness.pipeline.ingest(path, roles=frozenset({"legal"}), corpus="c", replace=True)

    assert new.removed_doc_ids == (old.doc_id,)
    assert await harness.documents.get(old.doc_id) is None
    assert await harness.vectors.count("c") == new.chunk_count


async def test_without_replace_both_versions_stay(harness: Harness, tmp_path: Path) -> None:
    path = write(tmp_path, "msa.txt", CONTRACT)
    old = await harness.pipeline.ingest(path, roles=frozenset({"legal"}), corpus="c")
    path.write_text(CONTRACT + "\n\nExtra clause.", encoding="utf-8")
    await harness.pipeline.ingest(path, roles=frozenset({"legal"}), corpus="c")

    assert await harness.documents.get(old.doc_id) is not None


async def test_empty_file_is_rejected_without_touching_storage(
    harness: Harness, tmp_path: Path
) -> None:
    with pytest.raises(EmptyDocumentError):
        await harness.pipeline.ingest(
            write(tmp_path, "empty.txt", "   \n"), roles=frozenset({"legal"}), corpus="c"
        )

    assert harness.embedder.calls == []
    assert await harness.vectors.count("c") == 0


async def test_simultaneous_ingests_of_one_file_end_in_one_complete_document(
    harness: Harness, tmp_path: Path
) -> None:
    path = write(
        tmp_path, "msa.txt", "\n\n".join(f"Clause {n}. " + "Text. " * 80 for n in range(8))
    )

    results = await asyncio.gather(
        *(harness.pipeline.ingest(path, roles=frozenset({"legal"}), corpus="c") for _ in range(3))
    )

    listed = await harness.documents.list_visible("c", frozenset({"legal"}), limit=10)
    assert len(listed) == 1
    assert await harness.vectors.count("c") == listed[0].chunk_count
    assert {r.doc_id for r in results} == {listed[0].id}
    assert len(await harness.documents.chunks(listed[0].id)) == listed[0].chunk_count
