import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from pydantic import SecretStr

from grounded_rag.errors import ConfigurationError, VectorStoreError
from grounded_rag.http_retry import Sleep, request_with_retries
from grounded_rag.vectorstore.base import VectorMatch, VectorRecord

CONTROL_PLANE_URL = "https://api.pinecone.io"
_UPSERT_BATCH = 100
_DELETE_BATCH = 1000
_READY_POLL_SECONDS = 2.0
_READY_ATTEMPTS = 60


class PineconeStore:
    """Pinecone serverless index over its REST API. Namespaces map to corpora."""

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        index_name: str,
        index_host: str | None,
        api_version: str,
        timeout_seconds: float,
        max_retries: int,
        control_url: str = CONTROL_PLANE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._index_host = index_host
        self._api_version = api_version
        self._timeout = httpx.Timeout(timeout_seconds)
        self._max_retries = max_retries
        self._transport = transport
        self._sleep = sleep
        self._control = httpx.AsyncClient(
            base_url=control_url, timeout=self._timeout, transport=transport
        )
        self._data: httpx.AsyncClient | None = None
        self._data_lock = asyncio.Lock()

    def _headers(self) -> dict[str, str]:
        if self._api_key is None:
            raise ConfigurationError("PINECONE_API_KEY is not set.")
        return {
            "Api-Key": self._api_key.get_secret_value(),
            "X-Pinecone-API-Version": self._api_version,
            "Accept": "application/json",
        }

    async def _call(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        *,
        accept_status: frozenset[int] = frozenset(),
    ) -> httpx.Response:
        return await request_with_retries(
            client,
            method,
            path,
            json=body,
            headers=self._headers(),
            provider="Pinecone",
            error=VectorStoreError,
            timeout_error=VectorStoreError,
            max_retries=self._max_retries,
            accept_status=accept_status,
            sleep=self._sleep,
        )

    async def _data_client(self) -> httpx.AsyncClient:
        async with self._data_lock:
            if self._data is None:
                host = self._index_host or await self._lookup_host()
                base_url = host if host.startswith("http") else f"https://{host}"
                self._data = httpx.AsyncClient(
                    base_url=base_url, timeout=self._timeout, transport=self._transport
                )
            return self._data

    async def _lookup_host(self) -> str:
        response = await self._call(
            self._control, "GET", f"/indexes/{self._index_name}", accept_status=frozenset({404})
        )
        if response.status_code == 404:
            raise VectorStoreError(
                f"Pinecone index '{self._index_name}' does not exist. "
                "Create it with `grounded-rag index create`."
            )
        return str(response.json()["host"])

    async def ensure_index(self, *, dimension: int, cloud: str, region: str) -> bool:
        """Create the serverless index if it is missing. Returns True when it was created."""
        response = await self._call(
            self._control,
            "POST",
            "/indexes",
            {
                "name": self._index_name,
                "dimension": dimension,
                "metric": "cosine",
                "spec": {"serverless": {"cloud": cloud, "region": region}},
            },
            accept_status=frozenset({409}),
        )
        await self._wait_until_ready()
        return response.status_code != 409

    async def _wait_until_ready(self) -> None:
        for _ in range(_READY_ATTEMPTS):
            response = await self._call(self._control, "GET", f"/indexes/{self._index_name}")
            if response.json().get("status", {}).get("ready"):
                return
            await self._sleep(_READY_POLL_SECONDS)
        raise VectorStoreError(f"Pinecone index '{self._index_name}' did not become ready in time.")

    async def upsert(self, corpus: str, records: Sequence[VectorRecord]) -> None:
        client = await self._data_client()
        for start in range(0, len(records), _UPSERT_BATCH):
            batch = records[start : start + _UPSERT_BATCH]
            await self._call(
                client,
                "POST",
                "/vectors/upsert",
                {"namespace": corpus, "vectors": [self._to_payload(record) for record in batch]},
            )

    @staticmethod
    def _to_payload(record: VectorRecord) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "doc_id": record.doc_id,
            "chunk_id": record.id,
            "allowed_roles": sorted(record.allowed_roles),
        }
        if record.section:
            metadata["section"] = record.section
        return {"id": record.id, "values": list(record.values), "metadata": metadata}

    async def query(
        self, corpus: str, vector: Sequence[float], *, top_k: int, roles: frozenset[str]
    ) -> list[VectorMatch]:
        if not roles:
            return []
        client = await self._data_client()
        response = await self._call(
            client,
            "POST",
            "/query",
            {
                "namespace": corpus,
                "vector": list(vector),
                "topK": top_k,
                "includeMetadata": False,
                "includeValues": False,
                "filter": {"allowed_roles": {"$in": sorted(roles)}},
            },
        )
        return [
            VectorMatch(id=match["id"], score=float(match["score"]))
            for match in response.json().get("matches", [])
        ]

    async def delete(self, corpus: str, ids: Sequence[str]) -> None:
        client = await self._data_client()
        for start in range(0, len(ids), _DELETE_BATCH):
            await self._call(
                client,
                "POST",
                "/vectors/delete",
                {"namespace": corpus, "ids": list(ids[start : start + _DELETE_BATCH])},
                accept_status=frozenset({404}),
            )

    async def count(self, corpus: str) -> int:
        client = await self._data_client()
        response = await self._call(client, "POST", "/describe_index_stats", {})
        namespaces = response.json().get("namespaces", {})
        return int(namespaces.get(corpus, {}).get("vectorCount", 0))

    async def aclose(self) -> None:
        await self._control.aclose()
        if self._data is not None:
            await self._data.aclose()
