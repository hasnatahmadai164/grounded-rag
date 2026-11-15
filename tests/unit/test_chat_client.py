import json

import httpx
import pytest
from pydantic import BaseModel, Field
from pydantic import SecretStr as Secret

from grounded_rag.errors import LLMOutputError
from grounded_rag.llm.client import (
    ChatMessage,
    ChatRequest,
    OpenAIChatClient,
    complete_structured,
    strict_json_schema,
)
from grounded_rag.llm.openai_http import OpenAIHttp
from grounded_rag.llm.pricing import Pricing

PRICING = Pricing(chat_input=1.0, chat_output=2.0, embedding=0.0)


def chat_body(content: str | None = '{"ok": true}', **choice: object) -> dict:
    message = {"role": "assistant", "content": content}
    message.update(choice.pop("message", {}))  # type: ignore[arg-type]
    return {
        "model": "model-x",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop", **choice}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def client_returning(body: dict, seen: list[httpx.Request] | None = None, temperature=0.0):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=body)

    http = OpenAIHttp(
        api_key=Secret("sk-test"),
        base_url="https://api.test/v1",
        timeout_seconds=5,
        max_retries=0,
        transport=httpx.MockTransport(handler),
    )
    return OpenAIChatClient(
        http,
        model="model-x",
        pricing=PRICING,
        temperature=temperature,
        default_max_output_tokens=256,
    )


def request(**overrides: object) -> ChatRequest:
    messages = (ChatMessage("system", "be brief"), ChatMessage("user", "hi"))
    return ChatRequest(messages=messages, **overrides)  # type: ignore[arg-type]


async def test_payload_carries_model_messages_limits_and_temperature() -> None:
    seen: list[httpx.Request] = []
    await client_returning(chat_body(), seen).complete(request())

    payload = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.test/v1/chat/completions"
    assert payload["model"] == "model-x"
    assert payload["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert payload["max_completion_tokens"] == 256
    assert payload["temperature"] == 0.0
    assert "response_format" not in payload


async def test_temperature_is_omitted_when_the_model_does_not_accept_it() -> None:
    seen: list[httpx.Request] = []
    await client_returning(chat_body(), seen, temperature=None).complete(request())
    assert "temperature" not in json.loads(seen[0].content)


async def test_structured_requests_ask_for_strict_json_schema() -> None:
    seen: list[httpx.Request] = []
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    await client_returning(chat_body(), seen).complete(
        request(schema_name="Verdict", json_schema=schema, max_output_tokens=64)
    )

    payload = json.loads(seen[0].content)
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Verdict", "schema": schema, "strict": True},
    }
    assert payload["max_completion_tokens"] == 64


async def test_result_reports_tokens_and_cost() -> None:
    result = await client_returning(chat_body("hello")).complete(request())

    assert result.content == "hello"
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (100, 20)
    assert result.usage.cost_usd == pytest.approx((100 * 1.0 + 20 * 2.0) / 1_000_000)
    assert result.cached is False


async def test_output_cut_off_at_the_token_limit_is_an_error() -> None:
    with pytest.raises(LLMOutputError, match="cut off"):
        await client_returning(chat_body(finish_reason="length")).complete(request())


async def test_a_refusal_is_reported_as_an_output_error() -> None:
    body = chat_body(None, message={"refusal": "I cannot help"})
    with pytest.raises(LLMOutputError, match="declined"):
        await client_returning(body).complete(request())


async def test_malformed_responses_are_reported_not_crashed_on() -> None:
    with pytest.raises(LLMOutputError):
        await client_returning({"choices": []}).complete(request())


class Verdict(BaseModel):
    sufficient: bool
    reason: str = Field(min_length=1, max_length=200)
    missing: list[str] = Field(default_factory=list, max_length=5)
    title: str = "x"


async def test_structured_completion_returns_a_validated_model() -> None:
    body = chat_body('{"sufficient": true, "reason": "ok", "missing": [], "title": "t"}')

    outcome = await complete_structured(
        client_returning(body), [ChatMessage("user", "q")], Verdict, purpose="grade"
    )

    assert outcome.value == Verdict(sufficient=True, reason="ok", missing=[], title="t")
    assert outcome.result.usage.prompt_tokens == 100


async def test_structured_completion_rejects_output_that_breaks_the_model() -> None:
    body = chat_body('{"sufficient": "maybe", "reason": ""}')

    with pytest.raises(LLMOutputError, match="Verdict"):
        await complete_structured(
            client_returning(body), [ChatMessage("user", "q")], Verdict, purpose="grade"
        )


async def test_structured_completion_rejects_text_that_is_not_json() -> None:
    with pytest.raises(LLMOutputError):
        await complete_structured(
            client_returning(chat_body("not json")),
            [ChatMessage("user", "q")],
            Verdict,
            purpose="grade",
        )


def test_strict_schema_requires_every_property_and_forbids_extras() -> None:
    schema = strict_json_schema(Verdict)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["sufficient", "reason", "missing", "title"]


def test_strict_schema_drops_keywords_the_api_rejects_but_keeps_field_names() -> None:
    schema = strict_json_schema(Verdict)

    assert "title" in schema["properties"]
    assert "minLength" not in schema["properties"]["reason"]
    assert "default" not in schema["properties"]["missing"]
    assert "maxItems" not in schema["properties"]["missing"]


class Inner(BaseModel):
    chunk_id: str
    quote: str


class Outer(BaseModel):
    items: list[Inner]
    note: str | None


def test_strict_schema_applies_to_nested_models() -> None:
    schema = strict_json_schema(Outer)

    inner = schema["$defs"]["Inner"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["chunk_id", "quote"]
    assert schema["required"] == ["items", "note"]
