import httpx
import pytest

from grounded_rag.api.app import create_app
from grounded_rag.api.middleware import resolve_request_id
from grounded_rag.config import Settings


@pytest.fixture
def client() -> httpx.AsyncClient:
    app = create_app(Settings(_env_file=None))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_healthz_reports_the_process_is_alive(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_response_carries_a_generated_request_id(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert len(response.headers["x-request-id"]) == 32


async def test_response_echoes_a_valid_inbound_request_id(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz", headers={"X-Request-ID": "trace-abc-123"})
    assert response.headers["x-request-id"] == "trace-abc-123"


def test_unsafe_inbound_request_id_is_replaced() -> None:
    assert resolve_request_id("bad id\nwith newline") != "bad id\nwith newline"
    assert resolve_request_id("short") != "short"
    assert resolve_request_id(None)
