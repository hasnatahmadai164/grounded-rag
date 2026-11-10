import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from grounded_rag.errors import (
    ConfigurationError,
    InvalidRequestError,
    LLMError,
    LLMOutputError,
    LLMTimeoutError,
)
from grounded_rag.ingestion.embedder import OpenAIEmbedder
from grounded_rag.llm.openai_http import OpenAIHttp
from grounded_rag.llm.pricing import Pricing

PRICING = Pricing(chat_input=0.15, chat_output=0.6, embedding=0.02)


class Recorder:
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def embedding_response(request: httpx.Request, dimensions: int = 3) -> httpx.Response:
    inputs = json.loads(request.content)["input"]
    data = [
        {"index": index, "embedding": [float(index)] * dimensions} for index in range(len(inputs))
    ]
    return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 10 * len(inputs)}})


def build(recorder: Recorder, *, key: str | None = "sk-test", retries: int = 2, batch: int = 96):
    http = OpenAIHttp(
        api_key=SecretStr(key) if key else None,
        base_url="https://api.test/v1",
        timeout_seconds=5,
        max_retries=retries,
        transport=httpx.MockTransport(recorder),
        sleep=recorder.sleep,
    )
    return OpenAIEmbedder(
        http, model="embed-model", dimensions=3, pricing=PRICING, batch_size=batch
    )


async def test_request_carries_model_dimensions_and_bearer_token() -> None:
    recorder = Recorder(embedding_response)
    await build(recorder).embed(["alpha", "beta"])

    request = recorder.requests[0]
    assert str(request.url) == "https://api.test/v1/embeddings"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert json.loads(request.content) == {
        "model": "embed-model",
        "input": ["alpha", "beta"],
        "dimensions": 3,
    }


async def test_usage_reports_tokens_and_cost() -> None:
    result = await build(Recorder(embedding_response)).embed(["a", "b", "c"])

    assert result.usage.embedding_tokens == 30
    assert result.usage.cost_usd == pytest.approx(30 * 0.02 / 1_000_000)


async def test_inputs_are_sent_in_batches_and_results_stay_in_order() -> None:
    recorder = Recorder(embedding_response)
    result = await build(recorder, batch=2).embed(["a", "b", "c", "d", "e"])

    assert [len(json.loads(r.content)["input"]) for r in recorder.requests] == [2, 2, 1]
    assert len(result.vectors) == 5


async def test_vectors_are_reordered_by_the_index_the_api_returns() -> None:
    def shuffled(request: httpx.Request) -> httpx.Response:
        data = [
            {"index": 1, "embedding": [1.0, 1.0, 1.0]},
            {"index": 0, "embedding": [0.0, 0.0, 0.0]},
        ]
        return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 2}})

    result = await build(Recorder(shuffled)).embed(["first", "second"])

    assert result.vectors == [[0.0] * 3, [1.0] * 3]


async def test_rate_limited_request_is_retried_after_the_advertised_delay() -> None:
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, headers={"retry-after": "2"}, json={"error": {"message": "slow"}}
            )
        return embedding_response(request)

    recorder = Recorder(flaky)
    result = await build(recorder).embed(["x"])

    assert len(result.vectors) == 1
    assert recorder.sleeps == [2.0]


async def test_gives_up_with_an_error_once_retries_are_exhausted() -> None:
    recorder = Recorder(lambda request: httpx.Response(503, json={"error": {"message": "down"}}))

    with pytest.raises(LLMError, match="503"):
        await build(recorder, retries=2).embed(["x"])

    assert len(recorder.requests) == 3
    assert len(recorder.sleeps) == 2


async def test_timeouts_are_retried_then_reported_as_timeouts() -> None:
    def hang(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    recorder = Recorder(hang)

    with pytest.raises(LLMTimeoutError):
        await build(recorder, retries=1).embed(["x"])

    assert len(recorder.requests) == 2


async def test_client_errors_are_not_retried_and_do_not_echo_the_key() -> None:
    recorder = Recorder(lambda request: httpx.Response(401, json={"error": {"message": "bad key"}}))

    with pytest.raises(LLMError) as raised:
        await build(recorder).embed(["x"])

    assert len(recorder.requests) == 1
    assert "sk-test" not in str(raised.value)


async def test_missing_api_key_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        await build(Recorder(embedding_response), key=None).embed(["x"])


async def test_unexpected_vector_dimension_is_rejected() -> None:
    recorder = Recorder(lambda request: embedding_response(request, dimensions=5))
    with pytest.raises(LLMOutputError):
        await build(recorder).embed(["x"])


async def test_empty_text_is_rejected_before_any_request() -> None:
    recorder = Recorder(embedding_response)
    with pytest.raises(InvalidRequestError):
        await build(recorder).embed(["fine", "  "])
    assert recorder.requests == []


async def test_no_inputs_means_no_request() -> None:
    recorder = Recorder(embedding_response)
    result = await build(recorder).embed([])
    assert result.vectors == []
    assert recorder.requests == []
