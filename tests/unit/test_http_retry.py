import httpx
import pytest

from grounded_rag.errors import LLMError, LLMTimeoutError
from grounded_rag.http_retry import MAX_BACKOFF_SECONDS, request_with_retries, retry_delay


async def no_sleep(_seconds: float) -> None:
    return None


def client_for(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="https://provider.test", transport=handler)


async def send(client: httpx.AsyncClient, **overrides: object) -> httpx.Response:
    options: dict[str, object] = {
        "provider": "Provider",
        "error": LLMError,
        "timeout_error": LLMTimeoutError,
        "max_retries": 1,
        "sleep": no_sleep,
    }
    return await request_with_retries(client, "POST", "/x", **(options | overrides))  # type: ignore[arg-type]


async def test_status_listed_as_acceptable_is_returned_instead_of_raised() -> None:
    client = client_for(httpx.MockTransport(lambda request: httpx.Response(404, json={})))
    response = await send(client, accept_status=frozenset({404}))
    assert response.status_code == 404


async def test_error_message_uses_the_providers_explanation() -> None:
    body = {"error": {"code": "INVALID_ARGUMENT", "message": "dimension mismatch"}}
    client = client_for(httpx.MockTransport(lambda request: httpx.Response(400, json=body)))
    with pytest.raises(LLMError, match="Provider returned 400: dimension mismatch"):
        await send(client)


async def test_flat_message_bodies_are_understood() -> None:
    client = client_for(
        httpx.MockTransport(lambda request: httpx.Response(422, json={"message": "nope"}))
    )
    with pytest.raises(LLMError, match="nope"):
        await send(client)


def test_retry_after_is_honoured_but_capped() -> None:
    assert retry_delay(0, "3") == 3.0
    assert retry_delay(0, "9999") == MAX_BACKOFF_SECONDS


def test_backoff_grows_with_each_attempt() -> None:
    assert retry_delay(3, None) > retry_delay(0, None)


def test_unparseable_retry_after_falls_back_to_backoff() -> None:
    assert 0.5 <= retry_delay(0, "soon") <= 0.75
