from grounded_rag.db.repositories import DocumentRepository
from grounded_rag.domain.documents import (
    Chunk,
    NewDocument,
    content_sha256,
    make_chunk_id,
    make_document_id,
)


async def seed_document(
    repository: DocumentRepository,
    *,
    title: str,
    chunk_texts: list[str],
    roles: frozenset[str] = frozenset({"legal"}),
    corpus: str = "default",
    ready: bool = True,
    sections: list[str | None] | None = None,
) -> NewDocument:
    text = "\n\n".join(chunk_texts)
    digest = content_sha256(f"{title}:{text}")
    document = NewDocument(
        id=make_document_id(corpus, digest),
        corpus=corpus,
        source=f"{title}.txt",
        title=title,
        content_hash=digest,
        allowed_roles=roles,
        text=text,
    )
    chunks: list[Chunk] = []
    position = 0
    for ordinal, chunk_text in enumerate(chunk_texts):
        section = sections[ordinal] if sections else None
        chunks.append(
            Chunk(
                id=make_chunk_id(document.id, ordinal),
                doc_id=document.id,
                ordinal=ordinal,
                text=chunk_text,
                start_char=position,
                end_char=position + len(chunk_text),
                section=section,
                context_header=f"{title} > {section}" if section else title,
            )
        )
        position += len(chunk_text) + 2
    await repository.save_pending(document, chunks)
    if ready:
        await repository.mark_ready(document.id)
    return document
