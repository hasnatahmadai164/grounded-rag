from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ValidationError

from grounded_rag.domain.usage import Usage
from grounded_rag.errors import LLMOutputError
from grounded_rag.llm.openai_http import OpenAIHttp
from grounded_rag.llm.pricing import Pricing

# Validation keywords that strict structured outputs do not accept. Pydantic still enforces
# them on the parsed result.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "default",
        "format",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "pattern",
        "title",
    }
)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    schema_name: str | None = None
    json_schema: Mapping[str, Any] | None = None
    max_output_tokens: int | None = None
    purpose: str = "chat"


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    usage: Usage
    latency_ms: float
    model: str
    cached: bool = False
    # Time the original call took beyond this replay. Zero unless the result came from a cache.
    replay_credit_ms: float = 0.0


class ChatClient(Protocol):
    async def complete(self, request: ChatRequest) -> ChatResult: ...


@dataclass(frozen=True, slots=True)
class Structured[T: BaseModel]:
    value: T
    result: ChatResult


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for `model` in the shape strict structured outputs require."""

    def convert(node: Any) -> Any:
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node
        converted: dict[str, Any] = {}
        for key, value in node.items():
            if key in _UNSUPPORTED_KEYWORDS:
                continue
            if key in ("properties", "$defs") and isinstance(value, dict):
                # Keys here are field or definition names, so only their values are converted.
                converted[key] = {name: convert(sub) for name, sub in value.items()}
            else:
                converted[key] = convert(value)
        if "properties" in converted:
            converted["additionalProperties"] = False
            converted["required"] = list(converted["properties"])
        return converted

    schema: dict[str, Any] = convert(model.model_json_schema())
    return schema


class OpenAIChatClient:
    def __init__(
        self,
        http: OpenAIHttp,
        *,
        model: str,
        pricing: Pricing,
        temperature: float | None,
        default_max_output_tokens: int,
    ) -> None:
        self._http = http
        self._model = model
        self._pricing = pricing
        self._temperature = temperature
        self._default_max_output_tokens = default_max_output_tokens

    async def complete(self, request: ChatRequest) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_completion_tokens": request.max_output_tokens or self._default_max_output_tokens,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if request.json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name or "response",
                    "schema": request.json_schema,
                    "strict": True,
                },
            }
        response = await self._http.post("/chat/completions", payload)
        return self._parse(response.body, response.latency_ms)

    def _parse(self, body: dict[str, Any], latency_ms: float) -> ChatResult:
        try:
            choice = body["choices"][0]
            message = choice["message"]
            usage = body["usage"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMOutputError("Chat response was missing expected fields.") from error
        if message.get("refusal"):
            raise LLMOutputError("The model declined to answer the request.")
        if choice.get("finish_reason") == "length":
            raise LLMOutputError("The model output was cut off at the token limit.")
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMOutputError("Chat response contained no text.")
        return ChatResult(
            content=content,
            usage=self._pricing.chat_usage(
                int(usage["prompt_tokens"]), int(usage["completion_tokens"])
            ),
            latency_ms=latency_ms,
            model=str(body.get("model", self._model)),
        )


def structured_request[T: BaseModel](
    messages: Sequence[ChatMessage],
    output: type[T],
    *,
    purpose: str,
    max_output_tokens: int | None = None,
) -> ChatRequest:
    return ChatRequest(
        messages=tuple(messages),
        schema_name=output.__name__,
        json_schema=strict_json_schema(output),
        max_output_tokens=max_output_tokens,
        purpose=purpose,
    )


def parse_structured[T: BaseModel](result: ChatResult, output: type[T]) -> T:
    try:
        return output.model_validate_json(result.content)
    except ValidationError as error:
        fields = sorted({".".join(str(part) for part in e["loc"]) for e in error.errors()})
        raise LLMOutputError(
            f"Model output did not match {output.__name__}: invalid fields {', '.join(fields)}."
        ) from error


async def complete_structured[T: BaseModel](
    client: ChatClient,
    messages: Sequence[ChatMessage],
    output: type[T],
    *,
    purpose: str,
    max_output_tokens: int | None = None,
) -> Structured[T]:
    """Ask for JSON matching `output` and return it validated."""
    request = structured_request(
        messages, output, purpose=purpose, max_output_tokens=max_output_tokens
    )
    result = await client.complete(request)
    return Structured(parse_structured(result, output), result)
