import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from grounded_rag.errors import ConfigurationError, VectorStoreError
from grounded_rag.vectorstore.base import VectorRecord
from grounded_rag.vectorstore.pinecone_store import PineconeStore

HOST = "grounded-rag-abc.svc.test.pinecone.io"


class FakePinecone:
    def __init__(self, handlers: dict[str, Callable[[httpx.Request], httpx.Response]]) -> None:
        self.requests: list[httpx.Request] = []
        self._handlers = handlers

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.host}{request.url.path}"
        return self._handlers[key](request)

    def bodies(self, path: str) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]


async def no_sleep(_seconds: float) -> None:
    return None


def store_for(fake: FakePinecone, *, key: str | None = "pc-test", host: str | None = None):
    return PineconeStore(
        api_key=SecretStr(key) if key else None,
        index_name="grounded-rag",
        index_host=host,
        api_version="2025-01",
        timeout_seconds=5,
        max_retries=1,
        transport=httpx.MockTransport(fake),
        sleep=no_sleep,
    )


def describe_index(ready: bool = True) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(
        200, json={"host": HOST, "status": {"ready": ready, "state": "Ready"}}
    )


def record(identifier: str, roles: set[str]) -> VectorRecord:
    return VectorRecord(
        id=identifier,
        values=[0.1, 0.2],
        doc_id="doc1",
        allowed_roles=frozenset(roles),
        section="Term",
    )


async def test_upsert_sends_only_ids_vectors_and_filterable_metadata() -> None:
    fake = FakePinecone(
        {
            "GET api.pinecone.io/indexes/grounded-rag": describe_index(),
            f"POST {HOST}/vectors/upsert": lambda r: httpx.Response(200, json={"upsertedCount": 1}),
        }
    )
    await store_for(fake).upsert("contracts", [record("doc1-00000", {"legal", "admin"})])

    body = fake.bodies("/vectors/upsert")[0]
    assert body["namespace"] == "contracts"
    assert body["vectors"] == [
        {
            "id": "doc1-00000",
            "values": [0.1, 0.2],
            "metadata": {
                "doc_id": "doc1",
                "chunk_id": "doc1-00000",
                "allowed_roles": ["admin", "legal"],
                "section": "Term",
            },
        }
    ]


async def test_requests_carry_the_api_key_and_version_headers() -> None:
    fake = FakePinecone(
        {
            "GET api.pinecone.io/indexes/grounded-rag": describe_index(),
            f"POST {HOST}/describe_index_stats": lambda r: httpx.Response(200, json={}),
        }
    )
    await store_for(fake).count("contracts")

    for request in fake.requests:
        assert request.headers["api-key"] == "pc-test"
        assert request.headers["x-pinecone-api-version"] == "2025-01"


async def test_upserts_are_sent_in_batches_of_one_hundred() -> None:
    fake = FakePinecone(
        {
            f"POST {HOST}/vectors/upsert": lambda r: httpx.Response(200, json={}),
        }
    )
    records = [record(f"doc1-{n:05d}", {"legal"}) for n in range(250)]

    await store_for(fake, host=HOST).upsert("c", records)

    assert [len(b["vectors"]) for b in fake.bodies("/vectors/upsert")] == [100, 100, 50]


async def test_query_filters_by_role_and_maps_matches() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"matches": [{"id": "a-00001", "score": 0.91}, {"id": "b-00002", "score": 0.5}]},
        )

    fake = FakePinecone({f"POST {HOST}/query": respond})

    matches = await store_for(fake, host=HOST).query(
        "c", [0.1, 0.2], top_k=5, roles=frozenset({"legal", "finance"})
    )

    body = fake.bodies("/query")[0]
    assert body["filter"] == {"allowed_roles": {"$in": ["finance", "legal"]}}
    assert body["topK"] == 5
    assert body["namespace"] == "c"
    assert [(m.id, m.score) for m in matches] == [("a-00001", 0.91), ("b-00002", 0.5)]


async def test_query_without_roles_never_reaches_the_service() -> None:
    fake = FakePinecone({})
    assert await store_for(fake, host=HOST).query("c", [0.1], top_k=5, roles=frozenset()) == []
    assert fake.requests == []


async def test_delete_ignores_a_namespace_that_no_longer_exists() -> None:
    fake = FakePinecone(
        {
            f"POST {HOST}/vectors/delete": lambda r: httpx.Response(
                404, json={"message": "Namespace not found"}
            )
        }
    )
    await store_for(fake, host=HOST).delete("c", ["a-00000"])


async def test_count_reads_the_namespace_vector_count() -> None:
    stats = {"namespaces": {"c": {"vectorCount": 42}, "other": {"vectorCount": 7}}}
    fake = FakePinecone(
        {f"POST {HOST}/describe_index_stats": lambda r: httpx.Response(200, json=stats)}
    )
    store = store_for(fake, host=HOST)

    assert await store.count("c") == 42
    assert await store.count("missing") == 0


async def test_data_host_is_looked_up_once_and_reused() -> None:
    fake = FakePinecone(
        {
            "GET api.pinecone.io/indexes/grounded-rag": describe_index(),
            f"POST {HOST}/describe_index_stats": lambda r: httpx.Response(200, json={}),
        }
    )
    store = store_for(fake)
    await store.count("c")
    await store.count("c")

    lookups = [r for r in fake.requests if r.url.host == "api.pinecone.io"]
    assert len(lookups) == 1


async def test_missing_index_points_the_operator_at_the_cli() -> None:
    fake = FakePinecone(
        {"GET api.pinecone.io/indexes/grounded-rag": lambda r: httpx.Response(404, json={})}
    )
    with pytest.raises(VectorStoreError, match="index create"):
        await store_for(fake).count("c")


async def test_service_errors_surface_as_vector_store_errors() -> None:
    fake = FakePinecone(
        {
            f"POST {HOST}/query": lambda r: httpx.Response(
                400, json={"error": {"message": "bad vector"}}
            )
        }
    )
    with pytest.raises(VectorStoreError, match="bad vector"):
        await store_for(fake, host=HOST).query("c", [0.1], top_k=1, roles=frozenset({"x"}))


async def test_missing_api_key_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        await store_for(FakePinecone({}), key=None, host=HOST).count("c")


async def test_ensure_index_creates_a_serverless_cosine_index_and_waits() -> None:
    state = {"polls": 0}

    def describe(request: httpx.Request) -> httpx.Response:
        state["polls"] += 1
        return httpx.Response(200, json={"host": HOST, "status": {"ready": state["polls"] > 1}})

    fake = FakePinecone(
        {
            "POST api.pinecone.io/indexes": lambda r: httpx.Response(
                201, json={"name": "grounded-rag"}
            ),
            "GET api.pinecone.io/indexes/grounded-rag": describe,
        }
    )

    created = await store_for(fake).ensure_index(dimension=1536, cloud="aws", region="us-east-1")

    assert created is True
    assert fake.bodies("/indexes")[0] == {
        "name": "grounded-rag",
        "dimension": 1536,
        "metric": "cosine",
        "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
    }
    assert state["polls"] == 2


async def test_ensure_index_reports_an_existing_index() -> None:
    fake = FakePinecone(
        {
            "POST api.pinecone.io/indexes": lambda r: httpx.Response(
                409, json={"error": {"message": "exists"}}
            ),
            "GET api.pinecone.io/indexes/grounded-rag": describe_index(),
        }
    )
    assert await store_for(fake).ensure_index(dimension=3, cloud="aws", region="us-east-1") is False
