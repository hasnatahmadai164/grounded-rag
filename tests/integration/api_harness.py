from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

from grounded_rag.api.app import create_app
from grounded_rag.config import Settings
from grounded_rag.container import Container, open_container
from grounded_rag.identity import generate_api_key
from tests.fakes import FakeEmbedder, InMemoryVectorStore, PurposeClient


@dataclass
class ApiHarness:
    """The real application and PostgreSQL, with fakes standing in for OpenAI and Pinecone."""

    client: httpx.AsyncClient
    container: Container
    chat: PurposeClient
    vectors: InMemoryVectorStore
    embedder: FakeEmbedder
    workdir: Path

    async def create_key(self, name: str, roles: set[str]) -> dict[str, str]:
        key = generate_api_key()
        await self.container.api_keys.create(
            name=name, key_hash=key.key_hash, key_prefix=key.key_prefix, roles=frozenset(roles)
        )
        return {"X-API-Key": key.secret}

    async def ingest(self, title: str, text: str, roles: set[str]) -> str:
        path = self.workdir / f"{title}.txt"
        path.write_text(text, encoding="utf-8")
        result = await self.container.ingestion.ingest(
            path, roles=frozenset(roles), corpus=self.container.settings.default_corpus
        )
        return result.doc_id


@asynccontextmanager
async def api_harness(
    database_url: str, workdir: Path, chat: PurposeClient | None = None, **overrides: object
) -> AsyncIterator[ApiHarness]:
    settings = Settings(
        _env_file=None, **({"database_url": database_url, "rerank_enabled": False} | overrides)
    )
    embedder, vectors = FakeEmbedder(), InMemoryVectorStore()
    chat = chat or PurposeClient()
    async with open_container(settings, embedder=embedder, vectors=vectors, chat=chat) as container:
        app = create_app(settings, container)
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield ApiHarness(client, container, chat, vectors, embedder, workdir)
