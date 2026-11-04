import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from tests.integration.db_utils import create_database, drop_database, run_alembic


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_live = pytest.mark.skip(reason="set RUN_LIVE=1 to run tests that call real services")
    skip_db = pytest.mark.skip(reason="set TEST_DATABASE_URL to run tests that need PostgreSQL")
    for item in items:
        if "live" in item.keywords and os.environ.get("RUN_LIVE") != "1":
            item.add_marker(skip_live)
        if "db" in item.keywords and not os.environ.get("TEST_DATABASE_URL"):
            item.add_marker(skip_db)


@pytest.fixture(autouse=True)
def reset_logging() -> Iterator[None]:
    yield
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


@pytest.fixture(scope="session")
def base_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    return url


@pytest.fixture(scope="session")
def migrated_database_url(base_database_url: str) -> Iterator[str]:
    """One migrated database per test session; tests start from empty tables."""
    url = asyncio.run(create_database(base_database_url))
    result = run_alembic(url, "upgrade", "head")
    assert result.returncode == 0, result.stderr
    try:
        yield url
    finally:
        asyncio.run(drop_database(url))


@pytest.fixture
async def engine(migrated_database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(migrated_database_url)
    async with engine.begin() as connection:
        await connection.execute(
            text("TRUNCATE documents, chunks, api_keys RESTART IDENTITY CASCADE")
        )
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
async def api(migrated_database_url: str, engine: AsyncEngine, tmp_path: Path):
    from tests.integration.api_harness import api_harness

    async with api_harness(migrated_database_url, tmp_path) as harness:
        yield harness
