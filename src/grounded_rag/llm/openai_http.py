import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import SecretStr

from grounded_rag.errors import ConfigurationError, LLMError, LLMTimeoutError
from grounded_rag.http_retry import Sleep, request_with_retries


@dataclass(frozen=True, slots=True)
class HttpResult:
    body: dict[str, Any]
    latency_ms: float


class OpenAIHttp:
    """JSON-over-HTTP access to the OpenAI API with timeouts and bounded retries."""

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._api_key = api_key
        self._max_retries = max_retries
        self._sleep = sleep
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def post(self, path: str, payload: Mapping[str, Any]) -> HttpResult:
        if self._api_key is None:
            raise ConfigurationError("OPENAI_API_KEY is not set.")
        started = time.perf_counter()
        response = await request_with_retries(
            self._client,
            "POST",
            path,
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
            provider="OpenAI",
            error=LLMError,
            timeout_error=LLMTimeoutError,
            max_retries=self._max_retries,
            sleep=self._sleep,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return HttpResult(body=response.json(), latency_ms=elapsed_ms)

    async def aclose(self) -> None:
        await self._client.aclose()
