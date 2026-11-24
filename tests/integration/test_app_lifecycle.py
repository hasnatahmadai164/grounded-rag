import httpx
import pytest

from grounded_rag.api.app import create_app
from grounded_rag.config import Settings

pytestmark = pytest.mark.db


async def test_the_app_opens_its_own_container_at_startup_and_serves_with_it(
    migrated_database_url: str, engine
) -> None:
    settings = Settings(_env_file=None, database_url=migrated_database_url)
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/v1/documents")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "missing_api_key"
