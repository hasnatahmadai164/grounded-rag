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
from tests.fakes import FakeEmbedder, InMemoryVectorStore

pytestmark = pytest.mark.db

runner = CliRunner()


@pytest.fixture
def stores(migrated_database_url: str, monkeypatch: pytest.MonkeyPatch):
    async def reset() -> None:
        engine = create_async_engine(migrated_database_url)
        async with engine.begin() as connection:
            await connection.execute(text("TRUNCATE documents, chunks CASCADE"))
        await engine.dispose()

    asyncio.run(reset())
    monkeypatch.setenv("DATABASE_URL", migrated_database_url)
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    embedder, vectors = FakeEmbedder(), InMemoryVectorStore()

    @asynccontextmanager
    async def fake_container(settings: Settings) -> AsyncIterator[Container]:
        async with open_container(settings, embedder=embedder, vectors=vectors) as container:
            yield container

    monkeypatch.setattr(cli, "open_container", fake_container)
    return vectors


def make_files(root: Path) -> None:
    (root / "a.txt").write_text("Alpha agreement governed by Delaware law.", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested" / "b.md").write_text("Beta agreement renews annually.", encoding="utf-8")
    (root / "ignored.docx").write_bytes(b"x")


def test_ingesting_a_directory_processes_every_supported_file(stores, tmp_path: Path) -> None:
    make_files(tmp_path)

    result = runner.invoke(cli.app, ["ingest", str(tmp_path), "--roles", "legal,finance"])

    assert result.exit_code == 0, result.output
    assert result.output.count("created") == 2
    assert asyncio.run(stores.count("default")) == 2


def test_running_ingest_twice_reports_unchanged(stores, tmp_path: Path) -> None:
    make_files(tmp_path)
    runner.invoke(cli.app, ["ingest", str(tmp_path), "--roles", "legal"])

    again = runner.invoke(cli.app, ["ingest", str(tmp_path), "--roles", "legal"])

    assert again.output.count("unchanged") == 2


def test_a_directory_without_documents_is_an_error(stores, tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["ingest", str(tmp_path), "--roles", "legal"])
    assert result.exit_code == 1
    assert "invalid_request" in result.output


def test_a_missing_path_is_an_error(stores, tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["ingest", str(tmp_path / "nope"), "--roles", "legal"])
    assert result.exit_code == 1


def test_delete_removes_an_ingested_document(stores, tmp_path: Path) -> None:
    make_files(tmp_path)
    ingested = runner.invoke(cli.app, ["ingest", str(tmp_path / "a.txt"), "--roles", "legal"])
    doc_id = ingested.output.split()[1]

    deleted = runner.invoke(cli.app, ["delete", doc_id])

    assert f"Deleted {doc_id}" in deleted.output
    assert asyncio.run(stores.count("default")) == 0


def test_title_is_rejected_for_several_files(stores, tmp_path: Path) -> None:
    make_files(tmp_path)
    result = runner.invoke(cli.app, ["ingest", str(tmp_path), "--roles", "legal", "--title", "X"])
    assert result.exit_code == 1
