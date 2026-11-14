from grounded_rag.domain.documents import make_chunk_id
from grounded_rag.retrieval.dense import DenseRetriever
from grounded_rag.vectorstore.base import VectorRecord
from tests.fakes import FakeEmbedder, InMemoryVectorStore, hashed_vector


async def store_with(*items: tuple[str, str, set[str]]) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.upsert(
        "c",
        [
            VectorRecord(
                id=make_chunk_id("doc", n),
                values=hashed_vector(text),
                doc_id="doc",
                allowed_roles=frozenset(roles),
            )
            for n, (_, text, roles) in enumerate(items)
        ],
    )
    return store


async def test_nearest_chunk_comes_first() -> None:
    store = await store_with(
        ("a", "governing law of delaware", {"legal"}),
        ("b", "payment due thirty days invoice", {"legal"}),
    )

    result = await DenseRetriever(FakeEmbedder(), store).search(
        "c", frozenset({"legal"}), "which law governs", limit=2
    )

    assert result.hits[0].chunk_id == "doc-00000"
    assert result.hits[0].score >= result.hits[1].score


async def test_role_filter_is_passed_to_the_vector_store() -> None:
    store = await store_with(
        ("a", "pricing schedule", {"finance"}),
        ("b", "pricing schedule", {"legal"}),
    )

    result = await DenseRetriever(FakeEmbedder(), store).search(
        "c", frozenset({"legal"}), "pricing schedule", limit=5
    )

    assert [h.chunk_id for h in result.hits] == ["doc-00001"]
    assert store.queries == [frozenset({"legal"})]


async def test_embedding_usage_is_reported() -> None:
    store = await store_with(("a", "text", {"legal"}))

    result = await DenseRetriever(FakeEmbedder(), store).search(
        "c", frozenset({"legal"}), "three word question", limit=1
    )

    assert result.usage.embedding_tokens == 3


async def test_empty_corpus_returns_no_hits() -> None:
    result = await DenseRetriever(FakeEmbedder(), InMemoryVectorStore()).search(
        "c", frozenset({"legal"}), "anything", limit=3
    )
    assert result.hits == []
