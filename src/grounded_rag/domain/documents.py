import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class DocumentStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    DELETING = "deleting"


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    corpus: str
    source: str
    title: str
    content_hash: str
    allowed_roles: frozenset[str]
    status: DocumentStatus
    char_count: int
    chunk_count: int
    created_at: datetime
    ready_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class NewDocument:
    """Everything known about a document before it has been stored."""

    id: str
    corpus: str
    source: str
    title: str
    content_hash: str
    allowed_roles: frozenset[str]
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A slice of a document. `text` is exactly `document_text[start_char:end_char]`."""

    id: str
    doc_id: str
    ordinal: int
    text: str
    start_char: int
    end_char: int
    section: str | None
    context_header: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk together with the document facts needed to authorise and present it."""

    chunk: Chunk
    document_title: str
    allowed_roles: frozenset[str]


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_document_id(corpus: str, content_hash: str) -> str:
    """Documents are content-addressed per corpus so re-ingesting the same text is a no-op."""
    return hashlib.sha256(f"{corpus}\0{content_hash}".encode()).hexdigest()[:32]


def make_chunk_id(doc_id: str, ordinal: int) -> str:
    return f"{doc_id}-{ordinal:05d}"
