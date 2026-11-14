import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grounded_rag.db.repositories import ChunkRepository, DocumentRepository
from grounded_rag.retrieval.lexical import lexical_search, query_terms
from tests.integration.seed import seed_document

pytestmark = pytest.mark.db

LEGAL = frozenset({"legal"})


@pytest.fixture
def documents(session_factory: async_sessionmaker[AsyncSession]) -> DocumentRepository:
    return DocumentRepository(session_factory)


@pytest.fixture
def chunks(session_factory: async_sessionmaker[AsyncSession]) -> ChunkRepository:
    return ChunkRepository(session_factory)


def test_query_terms_are_lowercase_distinct_words_in_order() -> None:
    assert query_terms("What is the Governing Law? Governing law!") == [
        "what",
        "is",
        "the",
        "governing",
        "law",
    ]


def test_query_terms_drop_punctuation_that_could_alter_a_tsquery() -> None:
    assert query_terms("a & b | !c ' ; drop") == ["a", "b", "c", "drop"]


def test_query_terms_are_capped() -> None:
    assert len(query_terms(" ".join(f"w{n}" for n in range(100)))) == 24


async def test_chunks_containing_the_query_words_are_found(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    doc = await seed_document(
        documents,
        title="Supply Agreement",
        chunk_texts=[
            "This agreement is governed by the laws of Delaware.",
            "Payment is due within thirty days.",
        ],
    )

    hits = await lexical_search(
        chunks, "default", LEGAL, "What law governs the agreement?", limit=5
    )

    assert hits[0].chunk_id == f"{doc.id}-00000"


async def test_chunks_matching_more_terms_rank_higher(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    doc = await seed_document(
        documents,
        title="Doc",
        chunk_texts=[
            "Audit rights are granted.",
            "The customer may audit books and records upon written notice.",
        ],
    )

    hits = await lexical_search(chunks, "default", LEGAL, "audit books records notice", limit=5)

    assert [h.chunk_id for h in hits] == [f"{doc.id}-00001", f"{doc.id}-00000"]


async def test_stemming_matches_different_word_forms(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    doc = await seed_document(
        documents, title="Doc", chunk_texts=["The term renews automatically."]
    )

    hits = await lexical_search(chunks, "default", LEGAL, "renewal", limit=5)

    assert [h.chunk_id for h in hits] == [f"{doc.id}-00000"]


async def test_the_document_title_is_searchable(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    doc = await seed_document(
        documents, title="Acme Distribution Agreement", chunk_texts=["Unrelated body text."]
    )

    hits = await lexical_search(chunks, "default", LEGAL, "Acme", limit=5)

    assert [h.chunk_id for h in hits] == [f"{doc.id}-00000"]


async def test_documents_the_roles_cannot_read_are_excluded(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    await seed_document(
        documents,
        title="Secret",
        chunk_texts=["Confidential pricing terms."],
        roles=frozenset({"finance"}),
    )

    assert await lexical_search(chunks, "default", LEGAL, "pricing", limit=5) == []
    assert (
        len(await lexical_search(chunks, "default", frozenset({"finance"}), "pricing", limit=5))
        == 1
    )


async def test_pending_documents_are_not_searched(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    await seed_document(documents, title="Draft", chunk_texts=["Pricing terms."], ready=False)

    assert await lexical_search(chunks, "default", LEGAL, "pricing", limit=5) == []


async def test_other_corpora_are_not_searched(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    await seed_document(documents, title="Doc", chunk_texts=["Pricing terms."], corpus="other")

    assert await lexical_search(chunks, "default", LEGAL, "pricing", limit=5) == []


async def test_no_roles_or_only_stop_words_return_nothing(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    await seed_document(documents, title="Doc", chunk_texts=["Pricing terms."])

    assert await lexical_search(chunks, "default", frozenset(), "pricing", limit=5) == []
    assert await lexical_search(chunks, "default", LEGAL, "the of and", limit=5) == []
    assert await lexical_search(chunks, "default", LEGAL, "???", limit=5) == []


async def test_limit_caps_the_number_of_hits(
    documents: DocumentRepository, chunks: ChunkRepository
) -> None:
    await seed_document(
        documents, title="Doc", chunk_texts=[f"Pricing clause {n}." for n in range(6)]
    )

    assert len(await lexical_search(chunks, "default", LEGAL, "pricing", limit=3)) == 3
