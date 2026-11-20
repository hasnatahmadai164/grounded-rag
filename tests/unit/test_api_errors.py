import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel, Field

from grounded_rag.api.errors import STATUS_BY_CODE, register_error_handlers
from grounded_rag.api.middleware import RequestContextMiddleware
from grounded_rag.errors import (
    GroundedRagError,
    InvalidApiKeyError,
    InvalidRequestError,
    LLMError,
    LLMTimeoutError,
    MissingApiKeyError,
    NotFoundError,
    VectorStoreError,
)


class Payload(BaseModel):
    question: str = Field(min_length=3)


def build_app() -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.add_middleware(RequestContextMiddleware)

    @app.get("/raise/{kind}")
    async def raise_error(kind: str) -> None:
        errors: dict[str, Exception] = {
            "missing": MissingApiKeyError("Provide X-API-Key."),
            "invalid": InvalidApiKeyError("Not valid."),
            "notfound": NotFoundError("No such thing."),
            "request": InvalidRequestError("top_k too large", details={"max": 20}),
            "llm": LLMError("OpenAI returned 500: internal host db-7.corp"),
            "timeout": LLMTimeoutError("took too long"),
            "vector": VectorStoreError("Pinecone returned 503 at secret-host"),
            "crash": RuntimeError("secret internals"),
        }
        raise errors[kind]

    @app.post("/echo")
    async def echo(payload: Payload) -> dict[str, str]:
        return {"question": payload.question}

    return app


@pytest.fixture
def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=build_app(), raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.parametrize(
    ("kind", "status", "code"),
    [
        ("missing", 401, "missing_api_key"),
        ("invalid", 401, "invalid_api_key"),
        ("notfound", 404, "not_found"),
        ("request", 422, "invalid_request"),
        ("llm", 502, "llm_error"),
        ("timeout", 504, "llm_timeout"),
        ("vector", 502, "vector_store_error"),
        ("crash", 500, "internal_error"),
    ],
)
async def test_each_failure_maps_to_one_status_and_a_stable_code(
    client: httpx.AsyncClient, kind: str, status: int, code: str
) -> None:
    response = await client.get(f"/raise/{kind}", headers={"X-Request-ID": "request-12345"})

    assert response.status_code == status
    error = response.json()["error"]
    assert error["code"] == code
    assert error["request_id"] == "request-12345"
    assert response.headers["x-request-id"] == "request-12345"


async def test_client_errors_keep_their_message_and_details(client: httpx.AsyncClient) -> None:
    error = (await client.get("/raise/request")).json()["error"]

    assert error["message"] == "top_k too large"
    assert error["details"] == {"max": 20}


@pytest.mark.parametrize("kind", ["llm", "vector", "crash"])
async def test_server_side_errors_do_not_leak_internal_text(
    client: httpx.AsyncClient, kind: str
) -> None:
    body = (await client.get(f"/raise/{kind}")).text

    assert "secret" not in body
    assert "db-7" not in body


async def test_validation_errors_name_the_field_but_not_the_submitted_value(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/echo", json={"question": "hi"})

    error = response.json()["error"]
    assert response.status_code == 422
    assert error["code"] == "invalid_request"
    assert error["details"]["fields"][0]["field"] == "body.question"
    assert '"hi"' not in response.text


async def test_unknown_routes_use_the_same_error_shape(client: httpx.AsyncClient) -> None:
    response = await client.get("/nowhere")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_wrong_methods_use_the_same_error_shape(client: httpx.AsyncClient) -> None:
    response = await client.delete("/echo")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_every_service_error_code_has_an_explicit_status() -> None:
    import inspect

    from grounded_rag import errors

    codes = {
        cls.code
        for _, cls in inspect.getmembers(errors, inspect.isclass)
        if issubclass(cls, GroundedRagError)
    }
    unmapped = codes - set(STATUS_BY_CODE)
    assert unmapped == set()
