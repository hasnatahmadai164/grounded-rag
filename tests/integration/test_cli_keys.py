import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from typer.testing import CliRunner

from grounded_rag.cli import app

pytestmark = pytest.mark.db

runner = CliRunner()


@pytest.fixture
def cli_database(migrated_database_url: str, monkeypatch: pytest.MonkeyPatch) -> str:
    async def reset() -> None:
        engine = create_async_engine(migrated_database_url)
        async with engine.begin() as connection:
            await connection.execute(text("TRUNCATE api_keys RESTART IDENTITY CASCADE"))
        await engine.dispose()

    asyncio.run(reset())
    monkeypatch.setenv("DATABASE_URL", migrated_database_url)
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    return migrated_database_url


def secret_from(output: str) -> str:
    line = next(line for line in output.splitlines() if line.startswith("API key"))
    return line.rsplit(" ", 1)[-1]


def test_created_key_is_printed_once_and_listed_by_prefix_only(cli_database: str) -> None:
    created = runner.invoke(
        app, ["keys", "create", "--name", "analyst", "--roles", "legal,finance"]
    )
    assert created.exit_code == 0, created.output
    secret = secret_from(created.output)

    listing = runner.invoke(app, ["keys", "list"])

    assert "analyst" in listing.output
    assert "finance,legal" in listing.output
    assert secret[:8] in listing.output
    assert secret not in listing.output


def test_revoked_key_is_marked_revoked(cli_database: str) -> None:
    runner.invoke(app, ["keys", "create", "--name", "temp", "--roles", "ops"])

    revoked = runner.invoke(app, ["keys", "revoke", "1"])
    listing = runner.invoke(app, ["keys", "list"])

    assert revoked.exit_code == 0
    assert "revoked" in listing.output


def test_revoking_an_unknown_key_fails(cli_database: str) -> None:
    result = runner.invoke(app, ["keys", "revoke", "99"])
    assert result.exit_code == 1


def test_invalid_role_names_are_rejected_with_a_stable_code(cli_database: str) -> None:
    result = runner.invoke(app, ["keys", "create", "--name", "bad", "--roles", "not valid"])
    assert result.exit_code == 1
    assert "invalid_request" in result.output
