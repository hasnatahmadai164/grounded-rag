import pytest

from tests.integration.db_utils import create_database, drop_database, run_alembic

pytestmark = pytest.mark.db


async def test_upgrade_creates_schema_matching_the_models(base_database_url: str) -> None:
    url = await create_database(base_database_url)
    try:
        assert run_alembic(url, "upgrade", "head").returncode == 0
        check = run_alembic(url, "check")
        assert check.returncode == 0, check.stdout + check.stderr
    finally:
        await drop_database(url)


async def test_downgrade_removes_every_table_and_upgrade_restores_them(
    base_database_url: str,
) -> None:
    url = await create_database(base_database_url)
    try:
        assert run_alembic(url, "upgrade", "head").returncode == 0
        assert run_alembic(url, "downgrade", "base").returncode == 0
        assert run_alembic(url, "upgrade", "head").returncode == 0
    finally:
        await drop_database(url)
