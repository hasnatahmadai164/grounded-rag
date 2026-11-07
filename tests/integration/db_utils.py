import os
import subprocess
import sys
import uuid
from pathlib import Path

import asyncpg
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[2]


def _admin_connect_args(url: str) -> dict[str, object]:
    parsed = make_url(url)
    return {
        "host": parsed.host,
        "port": parsed.port,
        "user": parsed.username,
        "password": parsed.password,
        "database": "postgres",
    }


async def create_database(base_url: str) -> str:
    name = f"grag_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(**_admin_connect_args(base_url))  # type: ignore[arg-type]
    try:
        await connection.execute(f'CREATE DATABASE "{name}"')
    finally:
        await connection.close()
    return make_url(base_url).set(database=name).render_as_string(hide_password=False)


async def drop_database(url: str) -> None:
    name = make_url(url).database
    connection = await asyncpg.connect(**_admin_connect_args(url))  # type: ignore[arg-type]
    try:
        await connection.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await connection.close()


def run_alembic(url: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=ROOT,
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        text=True,
        check=False,
    )
