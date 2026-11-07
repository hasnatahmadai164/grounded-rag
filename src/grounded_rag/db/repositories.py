from collections.abc import Sequence

from sqlalchemy import delete, desc, func, insert, literal_column, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grounded_rag.db.models import ApiKeyRow, ChunkRow, DocumentRow
from grounded_rag.domain.documents import (
    Chunk,
    Document,
    DocumentStatus,
    NewDocument,
    RetrievedChunk,
)
from grounded_rag.domain.principal import ApiKeyRecord
from grounded_rag.domain.retrieval import Hit


def _to_document(row: DocumentRow) -> Document:
    return Document(
        id=row.id,
        corpus=row.corpus,
        source=row.source,
        title=row.title,
        content_hash=row.content_hash,
        allowed_roles=frozenset(row.allowed_roles),
        status=DocumentStatus(row.status),
        char_count=row.char_count,
        chunk_count=row.chunk_count,
        created_at=row.created_at,
        ready_at=row.ready_at,
    )


def _to_chunk(row: ChunkRow) -> Chunk:
    return Chunk(
        id=row.id,
        doc_id=row.doc_id,
        ordinal=row.ordinal,
        text=row.text,
        start_char=row.start_char,
        end_char=row.end_char,
        section=row.section,
        context_header=row.context_header,
    )


class DocumentRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, doc_id: str) -> Document | None:
        async with self._sessions() as session:
            row = await session.get(DocumentRow, doc_id)
            return _to_document(row) if row else None

    async def get_text(self, doc_id: str) -> str | None:
        async with self._sessions() as session:
            return await session.scalar(select(DocumentRow.text).where(DocumentRow.id == doc_id))

    async def list_by_source(self, corpus: str, source: str) -> list[Document]:
        statement = (
            select(DocumentRow)
            .where(DocumentRow.corpus == corpus, DocumentRow.source == source)
            .order_by(DocumentRow.created_at, DocumentRow.id)
        )
        async with self._sessions() as session:
            return [_to_document(row) for row in await session.scalars(statement)]

    async def list_visible(
        self, corpus: str, roles: frozenset[str], *, limit: int, offset: int = 0
    ) -> list[Document]:
        statement = (
            select(DocumentRow)
            .where(
                DocumentRow.corpus == corpus,
                DocumentRow.status == DocumentStatus.READY,
                DocumentRow.allowed_roles.overlap(sorted(roles)),
            )
            .order_by(DocumentRow.created_at, DocumentRow.id)
            .limit(limit)
            .offset(offset)
        )
        async with self._sessions() as session:
            return [_to_document(row) for row in await session.scalars(statement)]

    async def save_pending(self, document: NewDocument, chunks: Sequence[Chunk]) -> bool:
        """Insert the document as `pending`, or refresh an unfinished one, with exactly `chunks`.

        A document that is already `ready` or `deleting` is left untouched and False is
        returned: a concurrent run finished it, or it is going away.
        """
        values = {
            "id": document.id,
            "corpus": document.corpus,
            "source": document.source,
            "title": document.title,
            "content_hash": document.content_hash,
            "text": document.text,
            "allowed_roles": sorted(document.allowed_roles),
            "status": DocumentStatus.PENDING,
            "char_count": len(document.text),
            "chunk_count": len(chunks),
        }
        insertion = pg_insert(DocumentRow).values(**values)
        upsert = insertion.on_conflict_do_update(
            index_elements=[DocumentRow.id],
            set_={key: insertion.excluded[key] for key in values if key != "id"},
            where=DocumentRow.status == DocumentStatus.PENDING,
        ).returning(DocumentRow.id)
        async with self._sessions.begin() as session:
            if await session.scalar(upsert) is None:
                return False
            await session.execute(delete(ChunkRow).where(ChunkRow.doc_id == document.id))
            if chunks:
                await session.execute(
                    insert(ChunkRow),
                    [
                        {
                            "id": chunk.id,
                            "doc_id": chunk.doc_id,
                            "ordinal": chunk.ordinal,
                            "text": chunk.text,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "section": chunk.section,
                            "context_header": chunk.context_header,
                        }
                        for chunk in chunks
                    ],
                )
        return True

    async def mark_ready(self, doc_id: str) -> bool:
        """Flip `pending` to `ready`. Returns False when the document is gone or being deleted."""
        statement = (
            update(DocumentRow)
            .where(DocumentRow.id == doc_id, DocumentRow.status == DocumentStatus.PENDING)
            .values(status=DocumentStatus.READY, ready_at=func.now())
        )
        async with self._sessions.begin() as session:
            result = await session.execute(statement)
            return bool(result.rowcount)  # type: ignore[attr-defined]

    async def mark_deleting(self, doc_id: str) -> bool:
        statement = (
            update(DocumentRow)
            .where(DocumentRow.id == doc_id)
            .values(status=DocumentStatus.DELETING)
        )
        async with self._sessions.begin() as session:
            result = await session.execute(statement)
            return bool(result.rowcount)  # type: ignore[attr-defined]

    async def update_roles(self, doc_id: str, roles: frozenset[str]) -> None:
        statement = (
            update(DocumentRow).where(DocumentRow.id == doc_id).values(allowed_roles=sorted(roles))
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def delete(self, doc_id: str) -> None:
        async with self._sessions.begin() as session:
            await session.execute(delete(DocumentRow).where(DocumentRow.id == doc_id))

    async def chunk_ids(self, doc_id: str) -> list[str]:
        statement = select(ChunkRow.id).where(ChunkRow.doc_id == doc_id).order_by(ChunkRow.ordinal)
        async with self._sessions() as session:
            return list(await session.scalars(statement))

    async def chunks(self, doc_id: str) -> list[Chunk]:
        statement = select(ChunkRow).where(ChunkRow.doc_id == doc_id).order_by(ChunkRow.ordinal)
        async with self._sessions() as session:
            return [_to_chunk(row) for row in await session.scalars(statement)]


def _to_api_key(row: ApiKeyRow) -> ApiKeyRecord:
    return ApiKeyRecord(
        id=row.id,
        name=row.name,
        key_prefix=row.key_prefix,
        roles=frozenset(row.roles),
        created_at=row.created_at,
        revoked_at=row.revoked_at,
    )


class ApiKeyRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(
        self, *, name: str, key_hash: str, key_prefix: str, roles: frozenset[str]
    ) -> ApiKeyRecord:
        row = ApiKeyRow(name=name, key_hash=key_hash, key_prefix=key_prefix, roles=sorted(roles))
        async with self._sessions.begin() as session:
            session.add(row)
            await session.flush()
            await session.refresh(row)
            return _to_api_key(row)

    async def find_active_by_hash(self, key_hash: str) -> ApiKeyRecord | None:
        statement = select(ApiKeyRow).where(
            ApiKeyRow.key_hash == key_hash, ApiKeyRow.revoked_at.is_(None)
        )
        async with self._sessions() as session:
            row = await session.scalar(statement)
            return _to_api_key(row) if row else None

    async def find_active_by_name(self, name: str) -> ApiKeyRecord | None:
        statement = (
            select(ApiKeyRow)
            .where(ApiKeyRow.name == name, ApiKeyRow.revoked_at.is_(None))
            .order_by(ApiKeyRow.id)
            .limit(1)
        )
        async with self._sessions() as session:
            row = await session.scalar(statement)
            return _to_api_key(row) if row else None

    async def list(self) -> list[ApiKeyRecord]:
        async with self._sessions() as session:
            rows = await session.scalars(select(ApiKeyRow).order_by(ApiKeyRow.id))
            return [_to_api_key(row) for row in rows]

    async def revoke(self, key_id: int) -> bool:
        """Revoke a key. Returns False when no active key has that id."""
        statement = (
            update(ApiKeyRow)
            .where(ApiKeyRow.id == key_id, ApiKeyRow.revoked_at.is_(None))
            .values(revoked_at=func.now())
        )
        async with self._sessions.begin() as session:
            result = await session.execute(statement)
            return bool(result.rowcount)  # type: ignore[attr-defined]


class ChunkRepository:
    """Read access to chunks for retrieval. Every query is scoped to ready documents the
    caller's roles may read."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def lexical_search(
        self, corpus: str, roles: frozenset[str], terms: Sequence[str], *, limit: int
    ) -> list[Hit]:
        """Rank chunks matching any of `terms`. Terms must already be plain lowercase words."""
        if not terms or not roles:
            return []
        query = func.to_tsquery(literal_column("'english'::regconfig"), " | ".join(terms))
        score = func.ts_rank_cd(ChunkRow.search_vector, query, 1)
        statement = (
            select(ChunkRow.id, score.label("score"))
            .join(DocumentRow, DocumentRow.id == ChunkRow.doc_id)
            .where(
                DocumentRow.corpus == corpus,
                DocumentRow.status == DocumentStatus.READY,
                DocumentRow.allowed_roles.overlap(sorted(roles)),
                ChunkRow.search_vector.op("@@")(query),
            )
            .order_by(desc("score"), ChunkRow.id)
            .limit(limit)
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return [Hit(chunk_id=row.id, score=float(row.score)) for row in result]

    async def visible_chunks(
        self, corpus: str, roles: frozenset[str], chunk_ids: Sequence[str]
    ) -> list[RetrievedChunk]:
        """The authoritative access check: which of these chunks may these roles read now?"""
        if not chunk_ids or not roles:
            return []
        statement = (
            select(ChunkRow, DocumentRow.title, DocumentRow.allowed_roles)
            .join(DocumentRow, DocumentRow.id == ChunkRow.doc_id)
            .where(
                ChunkRow.id.in_(list(chunk_ids)),
                DocumentRow.corpus == corpus,
                DocumentRow.status == DocumentStatus.READY,
                DocumentRow.allowed_roles.overlap(sorted(roles)),
            )
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return [
                RetrievedChunk(
                    chunk=_to_chunk(row),
                    document_title=title,
                    allowed_roles=frozenset(allowed_roles),
                )
                for row, title, allowed_roles in result
            ]
