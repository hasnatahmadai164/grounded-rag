import asyncio
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from grounded_rag.db.repositories import DocumentRepository
from grounded_rag.domain.documents import (
    Chunk,
    Document,
    DocumentStatus,
    NewDocument,
    content_sha256,
    make_chunk_id,
    make_document_id,
)
from grounded_rag.domain.usage import Usage
from grounded_rag.errors import IngestionError
from grounded_rag.ingestion.chunking import ChunkingConfig, chunk_text, make_context_header
from grounded_rag.ingestion.embedder import Embedder
from grounded_rag.ingestion.loaders import load_document
from grounded_rag.logging import get_logger
from grounded_rag.vectorstore.base import VectorRecord, VectorStore

log = get_logger(__name__)


class IngestOutcome(StrEnum):
    CREATED = "created"
    RESUMED = "resumed"
    UNCHANGED = "unchanged"
    ROLES_UPDATED = "roles_updated"


@dataclass(frozen=True, slots=True)
class IngestResult:
    doc_id: str
    title: str
    outcome: IngestOutcome
    chunk_count: int
    usage: Usage
    removed_doc_ids: tuple[str, ...] = ()


def build_chunks(doc_id: str, title: str, text: str, config: ChunkingConfig) -> list[Chunk]:
    return [
        Chunk(
            id=make_chunk_id(doc_id, ordinal),
            doc_id=doc_id,
            ordinal=ordinal,
            text=text[span.start : span.end],
            start_char=span.start,
            end_char=span.end,
            section=span.section,
            context_header=make_context_header(title, span.section),
        )
        for ordinal, span in enumerate(chunk_text(text, config))
    ]


def embedding_input(chunk: Chunk) -> str:
    """What gets embedded: the chunk text under its document and section names."""
    return f"{chunk.context_header}\n\n{chunk.text}"


class IngestionPipeline:
    """Writes to PostgreSQL first, then the vector store, and only then marks `ready`.

    Every step is keyed by content-derived ids, so re-running after a failure converges.
    """

    def __init__(
        self,
        documents: DocumentRepository,
        embedder: Embedder,
        vectors: VectorStore,
        chunking: ChunkingConfig,
    ) -> None:
        self._documents = documents
        self._embedder = embedder
        self._vectors = vectors
        self._chunking = chunking

    async def ingest(
        self,
        path: Path,
        *,
        roles: frozenset[str],
        corpus: str,
        title: str | None = None,
        replace: bool = False,
    ) -> IngestResult:
        """Ingest a file. With `replace`, older documents from the same source are removed after."""
        loaded = await asyncio.to_thread(load_document, path, title=title)
        doc_id = make_document_id(corpus, content_sha256(loaded.text))
        document = NewDocument(
            id=doc_id,
            corpus=corpus,
            source=loaded.source,
            title=loaded.title,
            content_hash=content_sha256(loaded.text),
            allowed_roles=roles,
            text=loaded.text,
        )
        existing = await self._documents.get(doc_id)
        if existing is not None and existing.status is DocumentStatus.DELETING:
            await self._purge(existing)
            existing = None

        if existing is not None and existing.status is DocumentStatus.READY:
            result = await self._reconcile_ready(existing, roles)
        else:
            result = await self._write(document, resumed=existing is not None)

        if replace:
            result = await self._remove_superseded(result, corpus, loaded.source)
        log.info(
            "document_ingested",
            doc_id=result.doc_id,
            outcome=result.outcome,
            chunks=result.chunk_count,
            embedding_tokens=result.usage.embedding_tokens,
            cost_usd=result.usage.cost_usd,
        )
        return result

    async def _write(self, document: NewDocument, *, resumed: bool) -> IngestResult:
        chunks = build_chunks(document.id, document.title, document.text, self._chunking)
        if not await self._documents.save_pending(document, chunks):
            return await self._finished_elsewhere(document.id)
        usage = await self._publish_vectors(document.corpus, chunks, document.allowed_roles)
        if not await self._documents.mark_ready(document.id):
            return await self._finished_elsewhere(document.id)
        outcome = IngestOutcome.RESUMED if resumed else IngestOutcome.CREATED
        return IngestResult(document.id, document.title, outcome, len(chunks), usage)

    async def _finished_elsewhere(self, doc_id: str) -> IngestResult:
        """Another run completed this document first, or it was deleted while we worked."""
        current = await self._documents.get(doc_id)
        if current is None or current.status is not DocumentStatus.READY:
            raise IngestionError(
                f"Document {doc_id} was removed while it was being ingested.",
                details={"doc_id": doc_id},
            )
        return IngestResult(
            current.id, current.title, IngestOutcome.UNCHANGED, current.chunk_count, Usage()
        )

    async def _reconcile_ready(self, existing: Document, roles: frozenset[str]) -> IngestResult:
        if existing.allowed_roles == roles:
            return IngestResult(
                existing.id, existing.title, IngestOutcome.UNCHANGED, existing.chunk_count, Usage()
            )
        # PostgreSQL is authoritative, so the new roles take effect before vectors are refreshed.
        await self._documents.update_roles(existing.id, roles)
        chunks = await self._documents.chunks(existing.id)
        usage = await self._publish_vectors(existing.corpus, chunks, roles)
        return IngestResult(
            existing.id, existing.title, IngestOutcome.ROLES_UPDATED, len(chunks), usage
        )

    async def _publish_vectors(
        self, corpus: str, chunks: list[Chunk], roles: frozenset[str]
    ) -> Usage:
        batch = await self._embedder.embed([embedding_input(chunk) for chunk in chunks])
        records = [
            VectorRecord(
                id=chunk.id,
                values=vector,
                doc_id=chunk.doc_id,
                allowed_roles=roles,
                section=chunk.section,
            )
            for chunk, vector in zip(chunks, batch.vectors, strict=True)
        ]
        await self._vectors.upsert(corpus, records)
        return batch.usage

    async def _remove_superseded(
        self, result: IngestResult, corpus: str, source: str
    ) -> IngestResult:
        stale = [
            d for d in await self._documents.list_by_source(corpus, source) if d.id != result.doc_id
        ]
        for document in stale:
            await self._purge(document)
        removed = tuple(document.id for document in stale)
        return IngestResult(
            result.doc_id,
            result.title,
            result.outcome,
            result.chunk_count,
            result.usage,
            removed_doc_ids=removed,
        )

    async def delete(self, doc_id: str) -> bool:
        """Remove a document. Safe to repeat: returns False when nothing is left to remove."""
        document = await self._documents.get(doc_id)
        if document is None:
            return False
        await self._purge(document)
        log.info("document_deleted", doc_id=doc_id)
        return True

    async def _purge(self, document: Document) -> None:
        # Marking first hides the document from retrieval even if a later step fails.
        await self._documents.mark_deleting(document.id)
        chunk_ids = await self._documents.chunk_ids(document.id)
        await self._vectors.delete(document.corpus, chunk_ids)
        await self._documents.delete(document.id)
