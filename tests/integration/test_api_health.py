import pytest
from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = pytest.mark.db


async def test_readyz_reports_ready_when_both_dependencies_answer(api) -> None:
    response = await api.client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"postgres": "ok", "pinecone": "ok"}}


async def test_readyz_needs_no_api_key(api) -> None:
    assert (await api.client.get("/readyz")).status_code == 200
    assert (await api.client.get("/healthz")).status_code == 200


async def test_readyz_fails_when_the_vector_store_is_unreachable(api) -> None:
    async def broken(corpus: str) -> int:
        raise ConnectionError("pinecone host db-9.internal unreachable")

    api.vectors.count = broken  # type: ignore[method-assign]

    response = await api.client.get("/readyz")

    error = response.json()["error"]
    assert response.status_code == 503
    assert error["code"] == "dependency_unavailable"
    assert error["details"]["checks"] == {"postgres": "ok", "pinecone": "unavailable"}
    assert "db-9" not in response.text


async def test_readyz_fails_when_postgres_is_unreachable(api) -> None:
    class BrokenSessions:
        def __call__(self) -> AsyncSession:
            raise ConnectionError("postgres down")

    api.container.sessions = BrokenSessions()  # type: ignore[assignment]

    response = await api.client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["error"]["details"]["checks"]["postgres"] == "unavailable"
