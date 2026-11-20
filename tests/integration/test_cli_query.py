import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from typer.testing import CliRunner

from grounded_rag import cli
from grounded_rag.config import Settings
from grounded_rag.container import Container, open_container
from tests.fakes import FakeEmbedder, InMemoryVectorStore, PurposeClient

pytestmark = pytest.mark.db

runner = CliRunner()
GOVERNING = "This Agreement shall be governed by the laws of the State of Delaware."
SUFFICIENT = {"sufficient": True, "reason": "found", "missing": ""}
ANSWER = {
    "answerable": True,
    "claims": [
        {
            "text": "Delaware law governs.",
            "citations": [{"passage_id": "P1", "quote": "the laws of the State of Delaware"}],
        }
    ],
}


@pytest.fixture
def setup(migrated_database_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    async def reset() -> None:
        engine = create_async_engine(migrated_database_url)
        async with engine.begin() as connection:
            await connection.execute(text("TRUNCATE documents, chunks, api_keys CASCADE"))
        await engine.dispose()

    asyncio.run(reset())
    monkeypatch.setenv("DATABASE_URL", migrated_database_url)
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("RERANK_ENABLED", "false")
    embedder, vectors = FakeEmbedder(), InMemoryVectorStore()
    chat = PurposeClient(grade=[SUFFICIENT, SUFFICIENT], generate=[ANSWER, ANSWER])

    @asynccontextmanager
    async def fake_container(settings: Settings) -> AsyncIterator[Container]:
        async with open_container(settings, embedder=embedder, vectors=vectors, chat=chat) as c:
            yield c

    monkeypatch.setattr(cli, "open_container", fake_container)
    document = tmp_path / "Supply Agreement.txt"
    document.write_text(f"13. Governing Law\n\n{GOVERNING}", encoding="utf-8")
    runner.invoke(cli.app, ["keys", "create", "--name", "analyst", "--roles", "legal"])
    runner.invoke(cli.app, ["keys", "create", "--name", "outsider", "--roles", "finance"])
    runner.invoke(cli.app, ["ingest", str(document), "--roles", "legal"])
    return chat


def test_a_principal_with_access_gets_a_cited_answer(setup: PurposeClient) -> None:
    result = runner.invoke(
        cli.app, ["query", "Which law governs the agreement?", "--as", "analyst"]
    )

    assert result.exit_code == 0, result.output
    assert "Delaware law governs. [1]" in result.output
    assert '"the laws of the State of Delaware"' in result.output
    assert "Supply Agreement, 13. Governing Law" in result.output
    assert "cost: $" in result.output


def test_a_principal_without_access_is_told_the_documents_do_not_answer(
    setup: PurposeClient,
) -> None:
    result = runner.invoke(
        cli.app, ["query", "Which law governs the agreement?", "--as", "outsider"]
    )

    assert result.exit_code == 0
    assert "abstained: no_relevant_context" in result.output
    assert "Delaware" not in result.output
    assert setup.requests == []


def test_an_unknown_principal_is_reported(setup: PurposeClient) -> None:
    result = runner.invoke(cli.app, ["query", "Anything?", "--as", "ghost"])

    assert result.exit_code == 1
    assert "not_found" in result.output


def test_debug_prints_the_retrieval_trace(setup: PurposeClient) -> None:
    result = runner.invoke(
        cli.app, ["query", "Which law governs the agreement?", "--as", "analyst", "--debug"]
    )

    assert "retrieval 1 fused" in result.output
    assert "grade:" in result.output
